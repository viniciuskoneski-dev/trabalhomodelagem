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

    # ----- Sets de pilhas novas e iniciais ------
    k_pile_initial_status = data["k_pile_initial_status"]
    NewPiles = {ps: [] for ps in PilePositions}
    InitialPiles = {ps: [] for ps in PilePositions}
    AllPiles = {ps: [] for ps in PilePositions}

    for ps in PilePositions:
        for p in Piles[ps]:
            if k_pile_initial_status[ps][p] == "NOVA":
                NewPiles[ps].append(p)
            else:
                InitialPiles[ps].append(p)
            AllPiles[ps].append(p)

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
    # 1. Quantidade de minério m fornecido pelo fornecedor f para a pilha p na posição ps no período t
    x = model.addVars(
        [(t, f, m, ps, p) 
         for t in range(Periods) 
         for f in Suppliers 
         for m in SupplierProducts.get(f, [])
         for ps in PilePositions
         for p in Piles[ps]], 
        name="x", vtype=GRB.CONTINUOUS, lb=0
    )
    
    # 2. Massa total da pilha p na posição ps no final do período t
    pile_mass = model.addVars(
        [(t, ps, p) 
         for t in range(Periods)
         for ps in PilePositions
         for p in Piles[ps]],
        name="pile_mass", vtype=GRB.CONTINUOUS, lb=0
    )

    # 3. Variáveis de desvio de qualidade (Falta e Sobra para o cálculo do alvo linearizado na Função Objetivo)
    dev_pos = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_pos", vtype=GRB.CONTINUOUS, lb=0
    )
    dev_neg = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_neg", vtype=GRB.CONTINUOUS, lb=0
    )

    # 4. Variável Binária de rota exclusiva (1 se a pilha p for alimentada apenas por trem)
    is_train = model.addVars(
        [(ps, p) for ps in PilePositions for p in Piles[ps]],
        name="is_train", vtype=GRB.BINARY
    )

    # 5. Variável de atendimento real da demanda (Sinterização)
    saida_real = model.addVars(
        [(t, ps, p) for t in range(Periods) for ps in PilePositions for p in Piles[ps]],
        name="saida_real", vtype=GRB.CONTINUOUS, lb=0
    )

    # ---- Restrições ----
    # 1. Balanceamento de massa da pilha
    for ps in PilePositions:
        for p in Piles[ps]:
            for t in range(Periods):
                entrada = gp.quicksum(x[t, f, m, ps, p] for f in Suppliers for m in SupplierProducts.get(f, []))
                saida_esperada = to_equip.get(ps, {}).get(p, [])[t] if ps in to_equip and p in to_equip[ps] and t < len(to_equip[ps][p]) else 0
                
                # Limita a saída real à demanda máxima exigida pelo equipamento
                model.addConstr(saida_real[t, ps, p] <= saida_esperada, name=f"max_demanda_{t}_{ps}_{p}")

                if t == 0:
                    massa_inicial = p_weight_init.get(ps, {}).get(p, 0)
                    model.addConstr(pile_mass[t, ps, p] == massa_inicial + entrada - saida_real[t, ps, p], name=f"bal_mass_0_{ps}_{p}")
                else:
                    model.addConstr(pile_mass[t, ps, p] == pile_mass[t-1, ps, p] + entrada - saida_real[t, ps, p], name=f"bal_mass_{t}_{ps}_{p}")

    # 2. Restrições de capacidade diária do fornecedor
    for t in range(Periods):
        for f in Suppliers:
            cap_f_t = data.get("m_supplier_delivery_capacity", {}).get(f, [])[t] if f in data.get("m_supplier_delivery_capacity", {}) else 0
            if cap_f_t > 0:
                model.addConstr(
                    gp.quicksum(x[t, f, m, ps, p] for m in SupplierProducts.get(f, []) for ps in PilePositions for p in Piles[ps]) <= cap_f_t,
                    name=f"cap_forn_{t}_{f}"
                )

    # 3. Restrição de movimentações máximas no pátio (diário total)
    for t in range(Periods):
        model.addConstr(
            gp.quicksum(x[t, f, m, ps, p] for f in Suppliers for m in SupplierProducts.get(f, []) for ps in PilePositions for p in Piles[ps]) <= daily_cap,
            name=f"cap_patio_{t}"
        )

    # 4. Respeito dos status das pilhas (nova, em construção, pronta)
    for ps in PilePositions:
        for p in Piles[ps]:
            for t in range(Periods):
                status = p_status.get(ps, {}).get(p, [])[t] if ps in p_status and p in p_status[ps] else "LIVRE"
                if status == "PRONTA":
                    # A pilha com status pronta não pode mais receber novas cargas de minério
                    model.addConstr(gp.quicksum(x[t, f, m, ps, p] for f in Suppliers for m in SupplierProducts.get(f, [])) == 0, name=f"status_pronta_{t}_{ps}_{p}")

    # 5. Restrição da alimentação exclusiva por trem ou rodoviário (Limitada pelas capacidades reais)
    for ps in PilePositions:
        for p in Piles[ps]:
            for t in range(Periods):
                for f in Suppliers:
                    # Capacidade diária do fornecedor via modal rodoviário
                    cap_rod_t = data.get("m_supplier_delivery_capacity", {}).get(f, [])[t] if f in data.get("m_supplier_delivery_capacity", {}) else 0
                    for m in SupplierProducts.get(f, []):
                        rota = mat_route.get(m, "")
                        if rota.upper() == "TREM":
                            cap_trem = train_cap.get(m, 0)
                            model.addConstr(x[t, f, m, ps, p] <= cap_trem * is_train[ps, p], name=f"train_excl_{t}_{f}_{m}_{ps}_{p}")
                        else:
                            model.addConstr(x[t, f, m, ps, p] <= cap_rod_t * (1 - is_train[ps, p]), name=f"road_excl_{t}_{f}_{m}_{ps}_{p}")

    # 6. Restrição de Balanço de Qualidade Alvo nas Pilhas
    for ps in PilePositions:
        for p in Piles[ps]:
            for q in QualityIndicators:
                alvo = target_qual.get(q, 0)
                
                # Calculando o total da qualidade acumulada com base nas massas de entradas
                massa_qual_entrada = gp.quicksum(x[t, f, m, ps, p] * qual.get(m, {}).get(q, 0) for t in range(Periods) for f in Suppliers for m in SupplierProducts.get(f, []))
                massa_total_entrada = gp.quicksum(x[t, f, m, ps, p] for t in range(Periods) for f in Suppliers for m in SupplierProducts.get(f, []))
                
                # Igualdade de massa idealizada: massa_adicionada * concentração = desvios tolerados
                model.addConstr(massa_qual_entrada - (massa_total_entrada * alvo) == dev_pos[ps, p, q] - dev_neg[ps, p, q], name=f"qual_bal_{ps}_{p}_{q}")

    # 7. Restrição de Disponibilidade do Produto (Limite de minério disponível na origem)
    for t in range(Periods):
        for m in Products:
            forns_m = [f for f in Suppliers if m in SupplierProducts.get(f, [])]
            if forns_m:
                # Captura a disponibilidade máxima do produto m no período t de forma robusta
                val_m = avail.get(m, 0)
                if isinstance(val_m, list):
                    disp_m_t = val_m[t] if t < len(val_m) else 0
                elif isinstance(val_m, dict):
                    disp_m_t = val_m.get(str(t), val_m.get(t, 0))
                else:
                    disp_m_t = val_m
                model.addConstr(
                    gp.quicksum(x[t, f, m, ps, p] for f in forns_m for ps in PilePositions for p in Piles[ps]) <= disp_m_t,
                    name=f"disp_prod_{t}_{m}"
                )

    # ----- Função objetivo -----
    # Minimizar as distâncias em relação à concentração alvo garantindo que a composição se aproxime da ideal 
    
    # Penalidade altíssima para demanda não atendida (garante que a Sinterização será atendida ao máximo possível)
    penalidade_falta = gp.quicksum(
        10000 * ( (to_equip.get(ps, {}).get(p, [])[t] if ps in to_equip and p in to_equip[ps] and t < len(to_equip[ps][p]) else 0) - saida_real[t, ps, p] )
        for t in range(Periods) for ps in PilePositions for p in Piles[ps]
    )

    model.setObjective(
        gp.quicksum(dev_pos[ps, p, q] + dev_neg[ps, p, q] for ps in PilePositions for p in Piles[ps] for q in QualityIndicators) + penalidade_falta,
        sense=GRB.MINIMIZE 
    )

    # ----- Solver params -----

    model.optimize()
    
    has_solution = model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0
    
    return has_solution, model, x, pile_mass, dev_pos, dev_neg, is_train, saida_real


# =========================================================================================================

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
        st.dataframe(df_avail, width="stretch")
        
    with col2:
        st.subheader("Qualidade dos Produtos")
        df_qual = pd.DataFrame(data["m_product_quality"]).T
        df_qual = df_qual * 1e2
        df_qual.columns = [f" {col} [%]" for col in df_qual.columns]
        st.dataframe(df_qual, width="stretch")



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
        st.dataframe(df_initial_weight, width="content")

    with col2:
        st.subheader("Condição Final das Pilhas")
        dic = []
        for pos, piles in data["m_pile_weight"].items():
            for p, mass in piles.items():
                record = {"Posição": pos, "Pilha": p, "Massa Final": mass}
                dic.append(record)
        df_final_weight = pd.DataFrame(dic).set_index(["Posição", "Pilha"])
        df_final_weight.columns = [f" {col} [kt]" for col in df_final_weight.columns]
        st.dataframe(df_final_weight, width="content")

    st.subheader("Status das Pilhas ao Longo do Tempo")
    status_records = []
    for pos, piles in data["k_pile_status"].items():
        for p, status_list in piles.items():
            record = {"Posição": pos, "Pilha": p}
            for t, st_val in enumerate(status_list):
                record[f"t{t+1}"] = st_val
            status_records.append(record)
    st.dataframe(pd.DataFrame(status_records).set_index(["Posição", "Pilha"]), width="stretch")


    ## -------------- Dados de Fornecedores --------------
    st.header("Dados de Fornecedores")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Relação Fornecedor-Produto")
        dic = []
        mat_route = data.get("k_material_route", {})
        for supplier, products in data["SupplierProducts"].items():
            for p in products:
                rota = mat_route.get(p, "RODOVIARIO")
                modo = "Trem" if rota.upper() == "TREM" else "Rodoviário"
                record = {"Fornecedor": supplier, "Produto": p, "Modal": modo}
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
    st.dataframe(df_daily_capacity, width="content")


elif aba_selecionada == "Código do Modelo":
    st.header("Instanciando o Modelo")
    st.markdown("Exemplo da estruturação dos conjuntos e parâmetros a partir dos dados de entrada:")
    
    code_snippet = '''

    def carregar_dados(filepath="data_piles.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

    data = carregar_dados()

    model = gp.Model("Pile_Case_UFMG")

    # ---- Sets -----
    Periods = 15
    Products = data["Products"]
    QualityIndicators = data["QualityIndicators"]
    Suppliers = data["Suppliers"]
    SupplierProducts = data["SupplierProducts"]
    PilePositions = data["PilePositions"]
    Piles = data["Piles"]

    # ---- Sugestão set de pilhas: novas e iniciais -----
    k_pile_initial_status = data["k_pile_initial_status"]
    NewPiles = {ps: [] for ps in PilePositions}
    InitialPiles = {ps: [] for ps in PilePositions}
    AllPiles = {ps: [] for ps in PilePositions}

    for ps in PilePositions:
        for p in Piles[ps]:
            if k_pile_initial_status[ps][p] == "NOVA":
                NewPiles[ps].append(p)
            else:
                InitialPiles[ps].append(p)
            AllPiles[ps].append(p)

    # ----- Parâmetros -----
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
    '''
    st.code(code_snippet, language='python')

elif aba_selecionada == "Resultados":
    st.header("Resultados do Modelo")
    
    if st.button("Executar Modelo", type="primary"):
        with st.spinner("Resolvendo modelo..."):
            has_solution, model, x, pile_mass, dev_pos, dev_neg, is_train, saida_real = gurobi_model(data)
            
            if has_solution:
                st.success(f"Solução ótima encontrada. OF: {model.ObjVal:.4f}")
                
                # --- VERIFICAÇÃO DE DEMANDA NÃO ATENDIDA ---
                unmet_records = []
                to_equip = data.get("m_to_equipment", {})
                for (t, ps, p), var in saida_real.items():
                    val = to_equip.get(ps, {}).get(p, 0)
                    if isinstance(val, list):
                        saida_esperada = val[t] if t < len(val) else 0
                    elif isinstance(val, dict):
                        saida_esperada = val.get(str(t), val.get(t, 0))
                    else:
                        saida_esperada = val
                        
                    if saida_esperada - var.X > 1e-4:
                        unmet_records.append({"Período": t+1, "Posição": ps, "Pilha": p, "Falta [kt]": saida_esperada - var.X})
                
                if unmet_records:
                    st.error("⚠️ Atenção: Parte da demanda da Sinterização não pôde ser atendida devido a limites logísticos ou falta de minério nas minas.")
                    st.dataframe(pd.DataFrame(unmet_records).style.format({"Falta [kt]": "{:.2f}"}), width="stretch")
                # -------------------------------------------

                # Extraindo os conjuntos do JSON para uso nos resultados do Streamlit
                Periods = 15
                PilePositions = data["PilePositions"]
                Piles = data["Piles"]
                QualityIndicators = data["QualityIndicators"]
                Suppliers = data["Suppliers"]
                SupplierProducts = data["SupplierProducts"]
                qual = data["m_product_quality"]
                
                tab_envios, tab_massas, tab_qualidade, tab_qual_tempo, tab_transporte = st.tabs([
                    "📦 Envios por Período", 
                    "⚖️ Evolução de Massas", 
                    "🎯 Desvios de Qualidade", 
                    "📈 Qualidade no Tempo",
                    "🚚 Transportes Realizados"
                ])
                
                with tab_envios:
                    st.subheader("Quantidade total enviada para cada pilha (kt)")
                    records_x = []
                    for (t, f, m, ps, p), var in x.items():
                        if var.X > 1e-4: # Apenas envios significativos (evita lixo de precisão do solver)
                            records_x.append({"Período": t + 1, "Posição": ps, "Pilha": p, "Fornecedor": f, "Produto": m, "Qtd [kt]": var.X})
                    
                    df_x = pd.DataFrame(records_x)
                    
                    if not df_x.empty:
                        # Agrupando por Pilha e Período (Pivot Table)
                        df_envios_pilha = df_x.groupby(["Posição", "Pilha", "Período"])["Qtd [kt]"].sum().reset_index()
                        df_pivot_envios = df_envios_pilha.pivot(index=["Posição", "Pilha"], columns="Período", values="Qtd [kt]").fillna(0)
                        # Renomeia as colunas inteiras de volta para texto após o pivot ordenar de forma numérica cronológica
                        df_pivot_envios.columns = [f"t{c}" for c in df_pivot_envios.columns]
                        st.dataframe(df_pivot_envios.style.format("{:.2f}"), width="stretch")
                        
                        st.caption("Detalhamento de Envios por Fornecedor e Produto:")
                        df_x.sort_values(by=["Período", "Posição", "Pilha"], inplace=True)
                        df_x["Período"] = df_x["Período"].apply(lambda t_val: f"t{t_val}")
                        st.dataframe(df_x, width="stretch")
                    else:
                        st.info("Nenhum envio foi realizado para as pilhas.")
                        
                with tab_massas:
                    st.subheader("Evolução da massa das pilhas ao final de cada período (kt)")
                    records_mass = []
                    for (t, ps, p), var in pile_mass.items():
                        records_mass.append({"Período": t + 1, "Posição": ps, "Pilha": p, "Massa [kt]": var.X})
                    
                    df_mass = pd.DataFrame(records_mass)
                    df_pivot_mass = df_mass.pivot(index=["Posição", "Pilha"], columns="Período", values="Massa [kt]").fillna(0)
                    df_pivot_mass.columns = [f"t{c}" for c in df_pivot_mass.columns]
                    st.dataframe(df_pivot_mass.style.format("{:.2f}"), width="stretch")
                    
                with tab_qualidade:
                    st.subheader("Desvios de Massa na Qualidade Alvo (Função Objetivo)")
                    st.markdown("Mostra o quanto a composição excedeu (Sobra) ou ficou abaixo (Falta) da massa ideal do composto para bater a meta de concentração.")
                    records_dev = []
                    for (ps, p, q), var in dev_pos.items():
                        if var.X > 1e-4 or dev_neg[ps, p, q].X > 1e-4:
                            records_dev.append({"Posição": ps, "Pilha": p, "Indicador": q, "Desvio Positivo (Sobra)": var.X, "Desvio Negativo (Falta)": dev_neg[ps, p, q].X})
                    
                    if records_dev:
                        st.dataframe(pd.DataFrame(records_dev).style.format({"Desvio Positivo (Sobra)": "{:.4f}", "Desvio Negativo (Falta)": "{:.4f}"}), width="stretch")
                    else:
                        st.success("🎉 O alvo de qualidade foi atingido com precisão máxima em todas as pilhas e indicadores!")
                        
                with tab_qual_tempo:
                    st.subheader("Evolução da Concentração de Qualidade (Adicionada) ao longo do tempo (%)")
                    st.markdown("Visualização da qualidade acumulada de material enviado para a pilha a cada período.")
                    
                    qual_records = []
                    for ps in PilePositions:
                        for p in Piles[ps]:
                            for q in QualityIndicators:
                                cum_mass = 0
                                cum_qual_mass = 0
                                for t in range(Periods):
                                    period_mass = sum(x[t, f, m, ps, p].X for f in Suppliers for m in SupplierProducts.get(f, []))
                                    period_qual = sum(x[t, f, m, ps, p].X * qual.get(m, {}).get(q, 0) for f in Suppliers for m in SupplierProducts.get(f, []))
                                    
                                    cum_mass += period_mass
                                    cum_qual_mass += period_qual
                                    
                                    if cum_mass > 1e-4: # Registra na tabela apenas se a pilha contiver massa
                                        conc = (cum_qual_mass / cum_mass) * 100
                                        qual_records.append({
                                            "Período": t + 1,
                                            "Posição": ps,
                                            "Pilha": p,
                                            "Indicador": q,
                                            "Concentração Acumulada [%]": conc
                                        })
                    
                    if qual_records:
                        df_qual_time = pd.DataFrame(qual_records)
                        df_qual_pivot = df_qual_time.pivot(index=["Posição", "Pilha", "Indicador"], columns="Período", values="Concentração Acumulada [%]").fillna(0)
                        df_qual_pivot.columns = [f"t{c}" for c in df_qual_pivot.columns]
                        st.dataframe(df_qual_pivot.style.format("{:.3f}"), width="stretch")
                    else:
                        st.info("Nenhuma massa foi adicionada às pilhas para calcular a evolução de qualidade.")

                with tab_transporte:
                    st.subheader("Transportes Realizados por Período")
                    st.markdown("Detalhamento de todos os transportes feitos, indicando origem, destino e quantidade transportada.")
                    mat_route = data["k_material_route"]
                    records_transporte = []
                    for (t, f, m, ps, p), var in x.items():
                        if var.X > 1e-4:
                            rota = mat_route.get(m, "RODOVIARIO")
                            modo = "🚂 Trem" if rota.upper() == "TREM" else "🚛 Rodoviário"
                            records_transporte.append({
                                "Período": t + 1,
                                "Fornecedor": f,
                                "Produto": m,
                                "Modal": modo,
                                "Posição Destino": ps,
                                "Pilha Destino": p,
                                "Qtd Transportada [kt]": var.X
                            })
                    
                    if records_transporte:
                        df_transporte = pd.DataFrame(records_transporte)
                        st.dataframe(df_transporte.style.format({"Qtd Transportada [kt]": "{:.2f}"}), hide_index=True, width="stretch")
                    else:
                        st.info("Nenhum transporte foi realizado no período.")

            else:
                st.error("Não foi possível encontrar solução viável.")
                
                # Se o modelo for inviável, calcula o IIS para descobrir o culpado
                if model.Status == GRB.INFEASIBLE:
                    with st.spinner("Calculando o núcleo de inviabilidade (IIS) para identificar o conflito..."):
                        model.computeIIS()
                        conflitos = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
                        
                        st.warning("⚠️ O modelo matemático é fisicamente impossível de ser atendido. As seguintes restrições estão em conflito direto:")
                        st.write(conflitos)
                        st.info("💡 Dica: Verifique se a demanda exigida pelo equipamento não é maior que a soma das capacidades dos fornecedores e disponibilidades no período.")