import streamlit as st
import pandas as pd
import json
import gurobipy as gp
from gurobipy import GRB
from streamlit_pdf_viewer import pdf_viewer

@st.cache_data
def carregar_dados(filepath="data_piles.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

data = carregar_dados()

def gurobi_model(data):
    
    model = gp.Model("Pile_Case_UFMG")

    # ----- Sets ------
    Periods = 15
    Products = data["Products"]
    QualityIndicators = data["QualityIndicators"]
    Suppliers = data["Suppliers"]
    SupplierProducts = data["SupplierProducts"]
    PilePositions = data["PilePositions"]
    Piles = data["Piles"]

    # ---- Param ------
    avail = data["m_product_delivery_availabity"]
    qual = data["m_product_quality"]
    train_cap = data["m_train_capacity"]
    mat_route = data["k_material_route"]
    p_status = data["k_pile_status"]
    p_weight_target = data["m_pile_weight"]
    p_weight_init = data["m_initial_pile_weight"]
    to_equip = data["m_to_equipment"]
    target_qual = data["m_target_quality"]
    daily_cap = data["m_dailyCapacity"]

    # --- Vars ---
    x = model.addVars(
        [(t, m, ps, p) for t in range(Periods) for m in Products for ps in PilePositions for p in Piles[ps]], 
        name="x", vtype=GRB.CONTINUOUS, lb=0
    )
    
    pile_mass = model.addVars(
        [(t, ps, p) for t in range(Periods) for ps in PilePositions for p in Piles[ps]],
        name="pile_mass", vtype=GRB.CONTINUOUS, lb=0
    )

    dev_pos = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_pos", vtype=GRB.CONTINUOUS, lb=0
    )
    
    dev_neg = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_neg", vtype=GRB.CONTINUOUS, lb=0
    )

    saida_real = model.addVars(
        [(t, ps, p) for t in range(Periods) for ps in PilePositions for p in Piles[ps]],
        name="saida_real", vtype=GRB.CONTINUOUS, lb=0
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

    # 3. Restrições de capacidade do trem por produto (Modais ferroviários)
    for t in range(Periods):
        for m in Products:
            if mat_route[m] == "Rail":
                cap_trem = train_cap[m]
                model.addConstr(
                    gp.quicksum(x[t, m, ps, p] for ps in PilePositions for p in Piles[ps]) <= cap_trem,
                    name=f"cap_trem_{t}_{m}"
                )

    # 4. Restrição de movimentações máximas no pátio (diário total)
    for t in range(Periods):
        model.addConstr(
            gp.quicksum(x[t, m, ps, p] for m in Products for ps in PilePositions for p in Piles[ps]) <= daily_cap,
            name=f"cap_patio_{t}"
        )

    # 5. Respeito dos status das pilhas (Revertido para a igualdade exata original)
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

    # 6. Restrição de Balanço de Qualidade Alvo (Loop ineficiente extraído)
    for ps in PilePositions:
        for p in Piles[ps]:
            massa_inicial = p_weight_init[ps][p]
            
            # Cálculo de massa posicionado acima do loop de qualidade (computado apenas 1x por pilha)
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

    # ----- Função objetivo Pura -----
    model.setObjective(
        gp.quicksum(dev_pos[ps, p, q] + dev_neg[ps, p, q] for ps in PilePositions for p in Piles[ps] for q in QualityIndicators),
        sense=GRB.MINIMIZE 
    )

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
        ["Material auxiliar", "Dados de Entrada", "Código do Modelo", "Resultados"]
    )
    
    st.divider()
    st.caption("Material desenvolvido e elaborado por Cassotis Consulting")


# tab_pdf, tab_input, tab_code, tab_output = st.tabs(["Material auxiliar", "Dados de Entrada", "Código do Modelo", "Resultados"])

if aba_selecionada == "Material auxiliar":
    st.header("Material Auxiliar")
    st.markdown("""
    Apresentação do case:
    """)

    pdf_viewer("Case_UFMG2026.pdf")

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

    # Criação das Abas Solicitadas
    aba1, aba2, aba3 = st.tabs(["Visão Geral", "Movimentações de Compra", "Status das Pilhas"])

    # --- ABA 1: VISÃO GERAL ---
    with aba1:
        st.header("Status do Problema")
        
        if st.button("Executar Otimização", type="primary", key="run_opt"):
            with st.spinner("Resolvendo modelo..."):
                # CORREÇÃO DA FALHA 1: Desempacotamento correto das variáveis
                has_solution, model, x, pile_mass, dev_pos, dev_neg, saida_real = gurobi_model(data)
                
                if has_solution:
                    st.success("✅ Solução ótima encontrada!")
                    st.metric(label="Valor da Função Objetivo (Minimização de Desvios)", value=f"{model.ObjVal:.4f}")
                    
                    # Salvando os resultados extraídos no session_state para persistência entre as abas
                    st.session_state['solved'] = True
                    st.session_state['x_vals'] = {(t, m, ps, p): x[t, m, ps, p].X for t, m, ps, p in x.keys()}
                    st.session_state['mass_vals'] = {(t, ps, p): pile_mass[t, ps, p].X for t, ps, p in pile_mass.keys()}
                else:
                    st.error("❌ Não foi possível encontrar solução viável para os dados fornecidos.")
                    st.session_state['solved'] = False
                    
        if 'solved' in st.session_state and st.session_state['solved']:
            st.info("Navegue pelas abas acima para explorar o detalhamento das compras e a composição dinâmica das pilhas.")

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
            
            # Filtrando apenas as movimentações que realmente ocorreram
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
                
                # Agrupamento exigido para mostrar o total por fornecedor/pilha no período
                df_grouped = df_movs.groupby(["Período", "Fornecedor", "Pilha de Destino"]).agg(
                    Total_Movimentado_kt=("Quantidade Movimentada (kt)", "sum"),
                    Minérios_Envolvidos=("Minério", lambda x: ", ".join(x))
                ).reset_index()
                
                st.dataframe(df_grouped, use_container_width=True, hide_index=True)
                
                with st.expander("Ver detalhamento granular por minério"):
                    st.dataframe(df_movs, use_container_width=True, hide_index=True)
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
            
            for ps in data["PilePositions"]:
                for p in data["Piles"][ps]:
                    st.subheader(f"{ps} - {p}")
                    
                    pile_data = []
                    # Acumulador para calcular as porcentagens dinâmicas
                    acc_ores = {m: 0.0 for m in data["Products"]}
                    massa_inicial = data["m_initial_pile_weight"][ps][p]
                    
                    for t in range(15): # 15 períodos definidos no modelo
                        status = data["k_pile_status"][ps][p][t]
                        current_mass = mass_vals[(t, ps, p)]
                        
                        # Atualiza o acumulador com os materiais que entraram na pilha
                        for m in data["Products"]:
                            acc_ores[m] += x_vals[(t, m, ps, p)]
                        
                        total_acc = sum(acc_ores.values()) + massa_inicial
                        
                        row = {
                            "Período": t + 1,
                            "Status": status,
                            "Massa Total (kt)": round(current_mass, 3)
                        }
                        
                        # Cálculo de porcentagem de cada minério adicionado até o momento
                        if total_acc > 1e-4:
                            if massa_inicial > 0:
                                row["% Massa Inicial"] = f"{(massa_inicial / total_acc) * 100:.1f}%"
                            for m, val in acc_ores.items():
                                if val > 1e-4:
                                    row[f"% {m}"] = f"{(val / total_acc) * 100:.1f}%"
                        else:
                            if massa_inicial > 0:
                                row["% Massa Inicial"] = "100.0%"
                            
                        pile_data.append(row)
                    
                    # Exibindo a tabela formatada, substituindo NaNs (minérios não usados) por vazios para ficar legível
                    df_pile = pd.DataFrame(pile_data).set_index("Período").fillna("-")
                    st.dataframe(df_pile, use_container_width=True)
                    st.divider()
        else:
            st.warning("Execute o modelo na aba 'Visão Geral' primeiro.")
