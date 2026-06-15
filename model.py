import streamlit as st
import pandas as pd
import json
import gurobipy as gp
from gurobipy import GRB
from streamlit_pdf_viewer import pdf_viewer
import os

@st.cache_data
def carregar_dados(filepath="data_piles.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

data = carregar_dados()

def gurobi_model(data, objective_type="1"):
    
    model = gp.Model("Pile_Case_UFMG")

    # ----- Sets ------
    Periods = 15
    Products = data["Products"]
    QualityIndicators = data["QualityIndicators"]
    Suppliers = data["Suppliers"]
    SupplierProducts = data["SupplierProducts"]
    PilePositions = data["PilePositions"]
    Piles = data["Piles"]
    
    # Filtrando quais produtos vem por trem
    mat_route = data["k_material_route"]
    RailProducts = [m for m in Products if mat_route[m] == "Rail"]

    # ---- Param ------
    avail = data["m_product_delivery_availabity"]
    qual = data["m_product_quality"]
    train_cap = data["m_train_capacity"]
    p_status = data["k_pile_status"]
    p_weight_target = data["m_pile_weight"]
    p_weight_init = data["m_initial_pile_weight"]
    to_equip = data["m_to_equipment"]
    target_qual = data["m_target_quality"]
    daily_cap = data["m_dailyCapacity"]

    # --- Vars ---
    # Quantidade de minério m adicionada à pilha p na posição ps durante o período t
    x = model.addVars(
        [(t, m, ps, p) for t in range(Periods) for m in Products for ps in PilePositions for p in Piles[ps]], 
        name="x", vtype=GRB.CONTINUOUS, lb=0
    )
    
    # Massa total acumulada  da pilha p na posição ps ao final do período t
    pile_mass = model.addVars(
        [(t, ps, p) for t in range(Periods) for ps in PilePositions for p in Piles[ps]],
        name="pile_mass", vtype=GRB.CONTINUOUS, lb=0
    )

    # Quantidade de minério  enviada da pilha p  para a Sinterização no período t
    saida_real = model.addVars(
        [(t, ps, p) for t in range(Periods) for ps in PilePositions for p in Piles[ps]],
        name="saida_real", vtype=GRB.CONTINUOUS, lb=0
    )
    
    # Variável de decisão binária que indica se o trem contendo o minério m foi utilizado no período t
    y_train = model.addVars(
        [(t, m) for t in range(Periods) for m in RailProducts],
        name="y_train", vtype=GRB.BINARY
    )


    # ---- Variaveis auxiliares ----
    # Desvio negativo em relação ao alvo do indicador de qualidade q na pilha p
    dev_neg = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_neg", vtype=GRB.CONTINUOUS, lb=0
    )

    # Desvio positivo em relação ao alvo do indicador de qualidade q na pilha p
    dev_pos = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_pos", vtype=GRB.CONTINUOUS, lb=0
    )


    # ---- Restrições ----
    # 1. Balanceamento de massa da pilha
    for ps in PilePositions:
        for p in Piles[ps]:
            for t in range(Periods):
                entrada = gp.quicksum(x[t, m, ps, p] for m in Products)
                saida_esperada = to_equip[ps][p][t]
                
                model.addConstr(saida_real[t, ps, p] == saida_esperada, name=f"demanda_exata_{t}_{ps}_{p}")

                if t == 0:
                    massa_inicial = p_weight_init[ps][p]
                    model.addConstr(pile_mass[t, ps, p] == massa_inicial + entrada - saida_real[t, ps, p], name=f"bal_mass_0_{ps}_{p}")
                else:
                    model.addConstr(pile_mass[t, ps, p] == pile_mass[t-1, ps, p] + entrada - saida_real[t, ps, p], name=f"bal_mass_{t}_{ps}_{p}")

    # 2. Restrições de capacidade diária do fornecedor (Modais rodoviários)
    for t in range(Periods):
        for f in Suppliers:
            cap_f_t = data["m_supplier_delivery_capacity"][f][t]
            prods_f = SupplierProducts[f]
            model.addConstr(
                gp.quicksum(x[t, m, ps, p] for m in prods_f for ps in PilePositions for p in Piles[ps]) <= cap_f_t,
                name=f"cap_forn_{t}_{f}"
            )

    # 3. Restrições de transporte ferroviário 
    for t in range(Periods):
        # 3.1 - Exclusividade: No máximo 1 trem chegando por dia
        model.addConstr(
            gp.quicksum(y_train[t, m] for m in RailProducts) <= 1,
            name=f"um_trem_exclusivo_dia_{t}"
        )
        
        # 3.2 - Restrição de quantidade
        for m in RailProducts:
            cap_trem = train_cap[m]
            model.addConstr(
                gp.quicksum(x[t, m, ps, p] for ps in PilePositions for p in Piles[ps]) <= cap_trem * y_train[t, m],
                name=f"cap_trem_{t}_{m}"
            )

    # 4. Restrição de movimentações máximas no pátio 
    for t in range(Periods):
        model.addConstr(
            gp.quicksum(x[t, m, ps, p] for m in Products for ps in PilePositions for p in Piles[ps]) <= daily_cap,
            name=f"cap_patio_{t}"
        )

    # 5. Respeito dos status das pilhas 
    for ps in PilePositions:
        for p in Piles[ps]:
            for t in range(Periods):
                status = p_status[ps][p][t]
                entrada = gp.quicksum(x[t, m, ps, p] for m in Products)
                
                if status == "VAZIA":
                    model.addConstr(entrada == 0, name=f"vazia_in_{t}_{ps}_{p}")
                    model.addConstr(saida_real[t, ps, p] == 0, name=f"vazia_out_{t}_{ps}_{p}")
                    model.addConstr(pile_mass[t, ps, p] == 0, name=f"vazia_mass_{t}_{ps}_{p}")
                    
                elif status == "CONSTRUCAO":
                    model.addConstr(saida_real[t, ps, p] == 0, name=f"const_out_{t}_{ps}_{p}")
                    
                elif status == "PRONTA":
                    model.addConstr(entrada == 0, name=f"pronta_in_{t}_{ps}_{p}")
                    model.addConstr(saida_real[t, ps, p] == 0, name=f"pronta_out_{t}_{ps}_{p}")
                    
                elif status == "CONSUMO":
                    model.addConstr(entrada == 0, name=f"consumo_in_{t}_{ps}_{p}")

    # 6. Restrição de Balanço de Qualidade Alvo
    for ps in PilePositions:
        for p in Piles[ps]:
            massa_inicial = p_weight_init[ps][p]
            
            massa_total_entrada = gp.quicksum(x[t, m, ps, p] for t in range(Periods) for m in Products)
            massa_total = massa_total_entrada + massa_inicial
            
            for q in QualityIndicators:
                alvo = target_qual[q]
                massa_qual_entrada = gp.quicksum(x[t, m, ps, p] * qual[m][q] for t in range(Periods) for m in Products)
                
                qual_total = massa_qual_entrada + (massa_inicial * alvo)
                
                model.addConstr(qual_total - (massa_total * alvo) == dev_pos[ps, p, q] - dev_neg[ps, p, q], name=f"qual_bal_{ps}_{p}_{q}")

    # 7. Restrição de Disponibilidade Máxima do Produto
    for m in Products:
        max_disp = avail[m]["max"]
        model.addConstr(
            gp.quicksum(x[t, m, ps, p] for t in range(Periods) for ps in PilePositions for p in Piles[ps]) <= max_disp,
            name=f"disp_prod_{m}"
        )

    # 8. Restrição de meta de massa final
    for ps in PilePositions:
        for p in Piles[ps]:
            if ps in p_weight_target and p in p_weight_target[ps]:
                target_w = p_weight_target[ps][p]
                model.addConstr(
                    gp.quicksum(x[t, m, ps, p] for t in range(Periods) for m in Products) == target_w,
                    name=f"target_weight_{ps}_{p}"
                )

    # ----- Seleção da Função Objetivo -----
    if objective_type == "1":
        # 1. Minimizar o erro em %
        model.setObjective(
            gp.quicksum((dev_pos[ps, p, q] + dev_neg[ps, p, q]) / (target_qual[q] if target_qual[q] > 0 else 1) 
                        for ps in PilePositions for p in Piles[ps] for q in QualityIndicators),
            sense=GRB.MINIMIZE 
        )
    
    elif objective_type == "2":
        # 2. Minimizar o erro absoluto bruto
        model.setObjective(
            gp.quicksum(dev_pos[ps, p, q] + dev_neg[ps, p, q] 
                        for ps in PilePositions for p in Piles[ps] for q in QualityIndicators),
            sense=GRB.MINIMIZE 
        )
        
    elif objective_type == "3":
        # 3. Minimizar a variância (Erro Quadrático Relativo)
        model.setObjective(
            gp.quicksum(
                (dev_pos[ps, p, q] * dev_pos[ps, p, q] + dev_neg[ps, p, q] * dev_neg[ps, p, q]) / 
                ((target_qual[q] ** 2) if target_qual[q] > 0 else 1)
                for ps in PilePositions for p in Piles[ps] for q in QualityIndicators
            ),
            sense=GRB.MINIMIZE 
        )
        
    elif objective_type == "4":
        # 4. Minimizar o maior desvio (Problema Min-Max)
        max_dev = model.addVar(name="max_dev", vtype=GRB.CONTINUOUS, lb=0)
        
        for ps in PilePositions:
            for p in Piles[ps]:
                for q in QualityIndicators:
                    target = target_qual[q] if target_qual[q] > 0 else 1
                    # Garante que o max_dev seja maior ou igual a qualquer desvio percentual de qualquer pilha
                    model.addConstr(max_dev >= (dev_pos[ps, p, q] + dev_neg[ps, p, q]) / target, name=f"minmax_{ps}_{p}_{q}")
                    
        model.setObjective(max_dev, sense=GRB.MINIMIZE)

    model.optimize()
    has_solution = model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0
    
    return has_solution, model, x, pile_mass, dev_pos, dev_neg, saida_real

    model.optimize()
    has_solution = model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0
    
    return has_solution, model, x, pile_mass, dev_pos, dev_neg, saida_real

## ------- STREAMLIT FRONT-END -------

st.set_page_config(layout="wide", page_title="Case Otimização UFMG")
st.logo(
    "side_bar_logo.png",
    link="https://www.cassotis.com/",
    icon_image="main_body_logo.png",
    size="large",
)
st.title("Case: Otimização da qualidade de pilhas de minério")
st.subheader("Cassotis Consulting - UFMG 2026.1")
st.divider()

with st.sidebar:
    st.title("Menu Principal")

    aba_selecionada = st.radio(
        "Ir para:",
        ["Relatório de explicação do código", "Dados de Entrada", "Código do Modelo", "Resultados"]
    )
    
    st.divider()
    st.caption("Material desenvolvido e elaborado por Cassotis Consulting")


# tab_pdf, tab_input, tab_code, tab_output = st.tabs(["Material auxiliar", "Dados de Entrada", "Código do Modelo", "Resultados"])

if aba_selecionada == "Relatório de explicação do código":
    st.header("Relatório de explicação do código")

    pdf_path = "Relatorio - entrega parcial.pdf"
    
    if os.path.exists(pdf_path):
        st.markdown("Relatório:")
        pdf_viewer(pdf_path)
    else:
        st.error(f"❌ Arquivo '{pdf_path}' não encontrado!")
        st.info("Certifique-se de que o PDF está salvo **exatamente com esse nome** e na mesma pasta em que o sistema está sendo executado.")
        st.code(f"Pasta atual onde o sistema está procurando: {os.path.abspath(os.getcwd())}")

elif aba_selecionada == "Dados de Entrada":

    ## -------------- Dados de Produtos --------------
    st.header("Dados de Produtos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Disponibilidade dos Produtos")
        df_avail = pd.DataFrame(data["m_product_delivery_availabity"]).T
        df_avail.columns = [f" {col} [kt]" for col in df_avail.columns]
        df_avail["Rota"] = pd.Series(data["k_material_route"])
        st.dataframe(df_avail, use_container_width=True)
        
    with col2:
        st.subheader("Qualidade dos Produtos")
        df_qual = pd.DataFrame(data["m_product_quality"]).T
        df_qual = df_qual * 1e2
        df_qual.columns = [f" {col} [%]" for col in df_qual.columns]
        st.dataframe(df_qual, use_container_width=True)



    ## -------------- Dados de Pilhas --------------
    st.header("Dados das Pilhas")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Condição Inicial das Pilhas")
        dic = []
        for pos, piles in data["m_initial_pile_weight"].items():
            for p, mass in piles.items():
                record = {"Posição": pos, "Pilha": p, "Massa Inicial": mass}
                dic.append(record)
        df_initial_weight = pd.DataFrame(dic).set_index(["Posição", "Pilha"])
        df_initial_weight.columns = [f" {col} [kt]" for col in df_initial_weight.columns]
        st.dataframe(df_initial_weight, use_container_width=False)

    with col2:
        st.subheader("Condição Final das Pilhas")
        dic = []
        for pos, piles in data["m_pile_weight"].items():
            for p, mass in piles.items():
                record = {"Posição": pos, "Pilha": p, "Massa Final": mass}
                dic.append(record)
        df_final_weight = pd.DataFrame(dic).set_index(["Posição", "Pilha"])
        df_final_weight.columns = [f" {col} [kt]" for col in df_final_weight.columns]
        st.dataframe(df_final_weight, use_container_width=False)

    st.subheader("Status das Pilhas ao Longo do Tempo")
    status_records = []
    for pos, piles in data["k_pile_status"].items():
        for p, status_list in piles.items():
            record = {"Posição": pos, "Pilha": p}
            for t, st_val in enumerate(status_list):
                record[f"t{t+1}"] = st_val
            status_records.append(record)
    st.dataframe(pd.DataFrame(status_records).set_index(["Posição", "Pilha"]), use_container_width=True)


    ## -------------- Dados de Fornecedores --------------
    st.header("Dados de Fornecedores")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Relação Fornecedor-Produto")
        dic = []
        for supplier, products in data["SupplierProducts"].items():
            for p in products:
                record = {"Fornecedor (rodoviário)": supplier, "Produto": p}
                dic.append(record)
        st.dataframe(pd.DataFrame(dic), hide_index=True)


    with col2:
        st.subheader("Capacidade do Trem por Produto")
        train_cap = []
        for products, capacity in data["m_train_capacity"].items():
            record = {"Produto": products, "Capacidade do trem [kt]": capacity}
            train_cap.append(record)
        df_train_cap = pd.DataFrame(train_cap)
        df_train_cap = df_train_cap[df_train_cap["Capacidade do trem [kt]"] != 0]
        st.dataframe(df_train_cap, hide_index=True)

    st.subheader("Capacidade máxima de entrega diária")
    dic = []
    for supplier, capacity_list in data["m_supplier_delivery_capacity"].items():
        record = {"Fornecedor": supplier}
        for t, val in enumerate(capacity_list):
            record[f"t{t+1}"] = val
        dic.append(record)
    df_supplier_delivery_capacity = pd.DataFrame(dic).set_index(["Fornecedor"])
    df_supplier_delivery_capacity.columns = [f" {col} [kt]" for col in df_supplier_delivery_capacity.columns]
    st.dataframe(df_supplier_delivery_capacity)

    ## -------------- Dados de Fornecedores --------------
    st.header("Dados do Equipamento")

    col1, col2 = st.columns(2)
    with col1:  
        st.subheader("Qualidade alvo")
        row = {"Equipamento": data["Equipments"][0], **data["m_target_quality"]}
        df_target_quality = pd.DataFrame([row]).set_index("Equipamento")
        df_target_quality = 1e2 * df_target_quality
        df_target_quality.columns = [f"{col} [%]" for col in df_target_quality.columns]
        st.dataframe(df_target_quality)

    st.subheader("Alimentação diária da Sinterização")
    demand = []
    for pos, piles in data["m_to_equipment"].items():
        for p, demand_list in piles.items():
            record = {"Posição": pos, "Pilha": p}
            for t, val in enumerate(demand_list):
                record[f"t{t+1}"] = val
            demand.append(record)
    df_demand = pd.DataFrame(demand).set_index(["Posição", "Pilha"])
    df_demand.columns = [f" {col} [kt]" for col in df_demand.columns]
    st.dataframe(df_demand)

    ## -------------- Dados do Pátio --------------
    st.header("Dados do Pátio")
    df_daily_capacity = pd.DataFrame([{"Pátio": "Pátio 1", "Capacidade max de movimentação diária [kt]": data["m_dailyCapacity"]}]).set_index("Pátio")
    st.dataframe(df_daily_capacity, use_container_width=False)


elif aba_selecionada == "Código do Modelo":
    st.header("Instanciando o Modelo")
    st.markdown("Exemplo da estruturação dos conjuntos e parâmetros a partir dos dados de entrada:")
    
    code_snippet = '''

    exemplo
    '''
    st.code(code_snippet, language='python')

elif aba_selecionada == "Resultados":
    st.header("Resultados do Modelo")
    st.divider()

    # Criação das 4 Abas Solicitadas
    aba1, aba2, aba3, aba4 = st.tabs(["Visão Geral", "Movimentações de Compra", "Status das Pilhas", "Sintetização"])

    # --- ABA 1: VISÃO GERAL ---
    with aba1:
        st.header("Status do Problema")
        
        st.markdown("**Selecione a abordagem de Função Objetivo para executar a Otimização:**")
        
        # Criação de um layout com 4 botões paralelos
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        run_1 = col_btn1.button("1. Minimizar Erro (%)", use_container_width=True, type="primary")
        run_2 = col_btn2.button("2. Minimizar Erro Absoluto", use_container_width=True, type="primary")
        run_3 = col_btn3.button("3. Minimizar Variância", use_container_width=True, type="primary")
        run_4 = col_btn4.button("4. Minimizar Maior Desvio", use_container_width=True, type="primary")
        
        # Define qual FO será rodada com base no clique
        fo_selecionada = None
        if run_1: fo_selecionada = "1"
        elif run_2: fo_selecionada = "2"
        elif run_3: fo_selecionada = "3"
        elif run_4: fo_selecionada = "4"
        
        if fo_selecionada:
            with st.spinner(f"Resolvendo modelo com a Função Objetivo {fo_selecionada}..."):
                has_solution, model, x, pile_mass, dev_pos, dev_neg, saida_real = gurobi_model(data, objective_type=fo_selecionada)
                
                if has_solution:
                    st.success("✅ Solução ótima encontrada!")
                    
                    # 1. Cálculo das médias de desvios relativos (%) por indicador de qualidade
                    medias_desvios_pct = {}
                    for q in data["QualityIndicators"]:
                        soma_desvio_q = 0.0
                        qtd_pilhas_avaliadas = 0
                        
                        for ps in data["PilePositions"]:
                            for p in data["Piles"][ps]:
                                massa_inicial = data["m_initial_pile_weight"][ps][p]
                                entrada_total = sum(x[t, m, ps, p].X for t in range(15) for m in data["Products"])
                                massa_base = massa_inicial + entrada_total
                                
                                if massa_base > 1e-4:
                                    alvo = data["m_target_quality"][q]
                                    q_mass_inicial = massa_inicial * alvo
                                    q_mass_adicionada = sum(x[t, m, ps, p].X * data["m_product_quality"][m][q] for t in range(15) for m in data["Products"])
                                    pct_q = (q_mass_inicial + q_mass_adicionada) / massa_base
                                    
                                    desvio_relativo = (abs(pct_q - alvo) / alvo) * 100
                                    soma_desvio_q += desvio_relativo
                                    qtd_pilhas_avaliadas += 1
                                    
                        medias_desvios_pct[q] = (soma_desvio_q / qtd_pilhas_avaliadas) if qtd_pilhas_avaliadas > 0 else 0.0
                    
                    # Salvando no session_state
                    st.session_state['solved'] = True
                    st.session_state['obj_val'] = model.ObjVal
                    st.session_state['fo_used'] = fo_selecionada
                    st.session_state['medias_desvios_pct'] = medias_desvios_pct
                    st.session_state['x_vals'] = {(t, m, ps, p): x[t, m, ps, p].X for t, m, ps, p in x.keys()}
                    st.session_state['mass_vals'] = {(t, ps, p): pile_mass[t, ps, p].X for t, ps, p in pile_mass.keys()}
                else:
                    st.error("❌ Não foi possível encontrar solução viável para os dados fornecidos.")
                    st.session_state['solved'] = False
                    
        # Exibição das métricas caso o modelo tenha sido executado com sucesso
        if 'solved' in st.session_state and st.session_state['solved']:
            st.divider()
            st.subheader("Métricas Operacionais Obtidas")
            
            # Mapenado nome dinâmico para a métrica de valor ótimo
            fo_name_map = {
                "1": "Erro Relativo (%)",
                "2": "Erro Absoluto (kt)",
                "3": "Variância Relativa (Erros Quadráticos)",
                "4": "Min-Max (Pior Desvio Relativo)"
            }
            fo_name = fo_name_map.get(st.session_state.get('fo_used', "1"), "Erro Relativo (%)")
            
            # Linha 1: Valor Ótimo com label dinâmico correspondente ao botão pressionado
            st.metric(label=f"Valor da Solução Ótima [F.O. - {fo_name}]", value=f"{st.session_state['obj_val']:.4f}")
            
            st.markdown("##### Média de Desvio por Indicador de Qualidade")
            st.caption("Representa o erro médio das pilhas em relação à qualidade alvo de 100%. Quanto mais próximo de zero, melhor.")
            
            # Linha 2: Desvios médios divididos pelas mineralógicas 
            col_d1, col_d2, col_d3 = st.columns(3)
            medias = st.session_state['medias_desvios_pct']
            
            with col_d1:
                st.metric(label="Média de Erro - FeT", value=f"{medias.get('FeT', 0):.2f}%")
            with col_d2:
                st.metric(label="Média de Erro - SiO2", value=f"{medias.get('SiO2', 0):.2f}%")
            with col_d3:
                st.metric(label="Média de Erro - Al2O3", value=f"{medias.get('Al2O3', 0):.2f}%")
            
            st.divider()
            st.info("Navegue pelas abas acima para explorar o detalhamento das compras, composição das pilhas e qualidade de sintetização geradas pela Função selecionada.")

    # --- ABA 2: MOVIMENTAÇÕES DE COMPRA ---
    with aba2:
        st.header("Movimentações por Período e Fornecedor")
        
        if 'solved' in st.session_state and st.session_state['solved']:
            # Mapeamento reverso para saber qual fornecedor entrega qual minério
            prod_to_sup = {}
            for sup, prods in data["SupplierProducts"].items():
                for pr in prods:
                    prod_to_sup[pr] = sup
            for pr in data["Products"]:
                if pr not in prod_to_sup:
                    prod_to_sup[pr] = "Ferrovia / Direto da Mina"
                    
            movs = []
            x_vals = st.session_state['x_vals']
            
            # Filtrando apenas as movimentações que realmente ocorreram (> 0)
            for (t, m, ps, p), val in x_vals.items():
                if val > 1e-4:
                    movs.append({
                        "Período": t + 1,
                        "Fornecedor": prod_to_sup[m],
                        "Pilha de Destino": f"{ps} - {p}",
                        "Minério": m,
                        "Quantidade Movimentada (kt)": round(val, 3)
                    })
            
            if movs:
                df_movs = pd.DataFrame(movs)
                
                # Gráfico de barras da quantidade de movimentações feitas por período
                st.subheader("Total Movimentado por Período")
                df_chart = df_movs.groupby("Período")["Quantidade Movimentada (kt)"].sum().reset_index()
                df_chart = df_chart.set_index("Período")
                st.bar_chart(df_chart)
                
                # Tabela de Movimentações (Agora exibindo linha por linha sem aglutinar minérios)
                st.subheader("Tabela de Movimentações")
                
                # Ordenação para manter a leitura fluida: primeiro pelo Período, depois por Fornecedor e Minério
                df_display = df_movs.sort_values(by=["Período", "Fornecedor", "Minério"])
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.warning("Nenhuma movimentação foi realizada pelo modelo.")
        else:
            st.warning("Execute o modelo na aba 'Visão Geral' primeiro.")

    # --- ABA 3: STATUS DAS PILHAS ---
    with aba3:
        st.header("Evolução de Massa e Composição por Pilha")
        
        if 'solved' in st.session_state and st.session_state['solved']:
            x_vals = st.session_state['x_vals']
            mass_vals = st.session_state['mass_vals']
            qual = data["m_product_quality"]
            target_qual = data["m_target_quality"]
            indicators = data["QualityIndicators"]
            
            for ps in data["PilePositions"]:
                for p in data["Piles"][ps]:
                    st.subheader(f"{ps} - {p}")
                    
                    pile_data = []
                    acc_ores = {m: 0.0 for m in data["Products"]}
                    massa_inicial = data["m_initial_pile_weight"][ps][p]
                    
                    for t in range(15):
                        status = data["k_pile_status"][ps][p][t]
                        current_mass = mass_vals[(t, ps, p)]
                        
                        for m in data["Products"]:
                            acc_ores[m] += x_vals[(t, m, ps, p)]
                        
                        total_base_mass = sum(acc_ores.values()) + massa_inicial
                        
                        pct_q = {}
                        for q in indicators:
                            if total_base_mass > 1e-4:
                                q_mass_inicial = massa_inicial * target_qual[q]
                                q_mass_adicionada = sum(acc_ores[m] * qual[m][q] for m in data["Products"])
                                pct_q[q] = (q_mass_inicial + q_mass_adicionada) / total_base_mass
                            else:
                                pct_q[q] = 0.0
                                
                        row = {
                            "Período": t + 1,
                            "Status": status,
                            "Massa total da pilha (kt)": round(current_mass, 3)
                        }
                        
                        for q in indicators:
                            if current_mass > 1e-4:
                                q_bruta = current_mass * pct_q[q]
                                row[f"Quantidade bruta de {q} (kt)"] = round(q_bruta, 3)
                                row[f"Quantidade em porcentagem de {q} (%)"] = f"{pct_q[q] * 100:.3f}%"
                            else:
                                row[f"Quantidade bruta de {q} (kt)"] = 0.0
                                row[f"Quantidade em porcentagem de {q} (%)"] = "-"
                                
                        pile_data.append(row)
                    
                    df_pile = pd.DataFrame(pile_data).set_index("Período")
                    
                    # NOVO: Gráfico de Linha da Evolução da Massa
                    st.line_chart(df_pile["Massa total da pilha (kt)"], color="#FF4B4B")
                    
                    # Tabela padronizada
                    st.dataframe(df_pile, use_container_width=True)
                    st.divider()
        else:
            st.warning("Execute o modelo na aba 'Visão Geral' primeiro.")

    # --- ABA 4: SINTETIZAÇÃO ---
    with aba4:
        st.header("Qualidade do Minério Enviado para Sintetização")
        
        if 'solved' in st.session_state and st.session_state['solved']:
            x_vals = st.session_state['x_vals']
            to_equip = data["m_to_equipment"]
            qual = data["m_product_quality"]
            target_qual = data["m_target_quality"]
            
            for ps in data["PilePositions"]:
                st.subheader(f"Posição: {ps}")
                sint_data = []
                
                # Listas para armazenar os dados normalizados dos gráficos
                chart_labels = []
                fet_real_norm, fet_alvo_norm = [], []
                sio2_real_norm, sio2_alvo_norm = [], []
                al2o3_real_norm, al2o3_alvo_norm = [], []
                
                # Percorrer o tempo para ver quais pilhas estão enviando massa para o equipamento
                for t in range(15):
                    for p in data["Piles"][ps]:
                        demanda = to_equip[ps][p][t]
                        
                        if demanda > 1e-4:
                            acc_ores = {m: 0.0 for m in data["Products"]}
                            massa_inicial = data["m_initial_pile_weight"][ps][p]
                            
                            for tau in range(t + 1): 
                                for m in data["Products"]:
                                    acc_ores[m] += x_vals[(tau, m, ps, p)]
                            
                            total_base_mass = sum(acc_ores.values()) + massa_inicial
                            
                            row = {"Período": t + 1, "Pilha de origem": p}
                            label = f"P{t+1} ({p})"
                            
                            if total_base_mass > 1e-4:
                                # ---------------- FeT ----------------
                                pct_fet = ((massa_inicial * target_qual["FeT"]) + sum(acc_ores[m] * qual[m]["FeT"] for m in data["Products"])) / total_base_mass
                                # Tabela mantém o valor bruto
                                row["Qualidade de FeT (%)"] = f"{pct_fet * 100:.3f}%"
                                # Gráfico recebe o valor normalizado em relação ao alvo
                                fet_real_norm.append((pct_fet / target_qual["FeT"]) * 100)
                                fet_alvo_norm.append(100.0)
                                
                                # ---------------- SiO2 ----------------
                                pct_sio2 = ((massa_inicial * target_qual["SiO2"]) + sum(acc_ores[m] * qual[m]["SiO2"] for m in data["Products"])) / total_base_mass
                                # Tabela mantém o valor bruto
                                row["Qualidade de SiO2 (%)"] = f"{pct_sio2 * 100:.3f}%"
                                # Gráfico recebe o valor normalizado em relação ao alvo
                                sio2_real_norm.append((pct_sio2 / target_qual["SiO2"]) * 100)
                                sio2_alvo_norm.append(100.0)
                                
                                # ---------------- Al2O3 ----------------
                                pct_al2o3 = ((massa_inicial * target_qual["Al2O3"]) + sum(acc_ores[m] * qual[m]["Al2O3"] for m in data["Products"])) / total_base_mass
                                # Tabela mantém o valor bruto
                                row["Qualidade de Al2O3 (%)"] = f"{pct_al2o3 * 100:.3f}%"
                                # Gráfico recebe o valor normalizado em relação ao alvo
                                al2o3_real_norm.append((pct_al2o3 / target_qual["Al2O3"]) * 100)
                                al2o3_alvo_norm.append(100.0)
                                
                                chart_labels.append(label)
                            else:
                                row["Qualidade de FeT (%)"] = "-"
                                row["Qualidade de SiO2 (%)"] = "-"
                                row["Qualidade de Al2O3 (%)"] = "-"
                                
                            sint_data.append(row)
                            
                if sint_data:
                    st.markdown("**Comparativo Relativo: Real vs Alvo (Normalizado para 100%)**")
                    st.caption("A barra escura representa a meta travada em 100%. A barra clara mostra o percentual de desvio atingido.")
                    
                    # Dividindo os gráficos em 3 colunas para um layout mais limpo
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**FeT (Proporção do Alvo)**")
                        df_fet = pd.DataFrame({"Real (% do Alvo)": fet_real_norm, "Meta (100%)": fet_alvo_norm}, index=chart_labels)
                        st.bar_chart(df_fet, color=["#87CEEB", "#00008B"], height=300, stack=False)
                        
                    with col2:
                        st.markdown("**SiO2 (Proporção do Alvo)**")
                        df_sio2 = pd.DataFrame({"Real (% do Alvo)": sio2_real_norm, "Meta (100%)": sio2_alvo_norm}, index=chart_labels)
                        st.bar_chart(df_sio2, color=["#FF7F7F", "#8B0000"], height=300, stack=False)
                        
                    with col3:
                        st.markdown("**Al2O3 (Proporção do Alvo)**")
                        df_al2o3 = pd.DataFrame({"Real (% do Alvo)": al2o3_real_norm, "Meta (100%)": al2o3_alvo_norm}, index=chart_labels)
                        st.bar_chart(df_al2o3, color=["#90EE90", "#006400"], height=300, stack=False)
                    
                    # Exibição da tabela de dados estruturada abaixo dos gráficos
                    st.subheader("Detalhamento da Qualidade (Valores Absolutos)")
                    df_sint = pd.DataFrame(sint_data).set_index("Período")
                    st.dataframe(df_sint, use_container_width=True)
                else:
                    st.info(f"Nenhuma sintetização programada para as pilhas pertencentes à {ps}.")
                st.divider()
        else:
            st.warning("Execute o modelo na aba 'Visão Geral' primeiro.")
