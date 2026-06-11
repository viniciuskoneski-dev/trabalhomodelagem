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
    m_pile_initial_weight = data["m_initial_pile_weight"]

    # --- Vars ---
    # 1. Quantidade de minério m enviado para a pilha p na posição ps no período t
    x = model.addVars(
        [(t, m, ps, p) 
         for t in range(Periods) 
         for m in Products
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

    # 3. Variáveis de desvio de qualidade (Falta e Sobra)
    dev_pos = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_pos", vtype=GRB.CONTINUOUS, lb=0
    )
    dev_neg = model.addVars(
        [(ps, p, q) for ps in PilePositions for p in Piles[ps] for q in QualityIndicators],
        name="dev_neg", vtype=GRB.CONTINUOUS, lb=0
    )

    # 4. Variável de atendimento real da demanda (Sinterização)
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

   # 6. Restrição de Balanço de Qualidade Alvo (Com correção da Massa Inicial)
    for ps in PilePositions:
        for p in Piles[ps]:
            massa_inicial = p_weight_init[ps][p]
            
            for q in QualityIndicators:
                alvo = target_qual[q]
                
                # Material adicionado (x)
                massa_qual_entrada = gp.quicksum(x[t, m, ps, p] * qual[m][q] for t in range(Periods) for m in Products)
                massa_total_entrada = gp.quicksum(x[t, m, ps, p] for t in range(Periods) for m in Products)
                
                # Equação Final: Adicionamos a Massa Inicial assumindo que ela possui qualidade equivalente ao 'alvo'
                qual_total = massa_qual_entrada + (massa_inicial * alvo)
                massa_total = massa_total_entrada + massa_inicial
                
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

    # ----- Função objetivo -----
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
    
    if st.button("Executar Modelo", type="primary"):
        with st.spinner("Resolvendo modelo..."):
            has_solution, model = gurobi_model(data)
            
            if has_solution:
                st.success(f"Solução ótima encontrada. OF: {model.ObjVal:.4f}")
                
            else:
                st.error("Não foi possível encontrar solução viável.")
