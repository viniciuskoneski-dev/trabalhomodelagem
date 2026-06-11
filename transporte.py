import gurobipy as gp

# Parâmetros do problema
qtd_fabricas = 4
qtd_clientes = 9
vet_ofertas = [140, 160, 200, 190]
vet_demandas = [50, 80, 30, 50, 100, 90, 20, 70, 120] #Oferta>demanda
vet_custos = [[12, 25, 39, 17, 38, 40, 8, 25, 13],
              [17, 26, 20, 25, 30, 25, 14, 20, 15],
              [35, 15, 18, 20, 12, 42, 27, 26, 19],
              [28, 30, 37, 30, 28, 36, 16, 24, 32]]
oferta_total = sum(vet_ofertas)
demanda_total = sum(vet_demandas)

# Rótulos das fábricas e clientes
fabricas = list()
for i in range(qtd_fabricas):
    fabricas.append("Fab_{}".format(i + 1))

clientes = list()
for j in range(qtd_clientes):
    clientes.append("Cli_{}".format(j + 1))

# Dicionários com as ofertas
ofertas = dict()
for idx, valor in enumerate(vet_ofertas):
    rotulo = fabricas[idx]
    ofertas[rotulo] = valor

# Dicionários com as demandas
demandas= dict()
for idx, valor in enumerate(vet_demandas):
    rotulo = clientes[idx]
    demandas[rotulo] = valor

custos = dict()
for i in range(qtd_fabricas):
    for j in range(qtd_clientes):
        rot_fabricas = fabricas[i]
        rot_clientes = clientes[j]
        custos[rot_fabricas, rot_clientes] = vet_custos[i][j]

# Criar o modelo
m =gp.Model()

# Criar as variáveis de decisão
x = m.addVars(fabricas, clientes, vtype=gp.GRB.INTEGER)

# Criar a função objetivo
m.setObjective(
    gp.quicksum(x[i, j] * custos[i, j] for i in fabricas for j in clientes),
    sense=gp.GRB.MINIMIZE
)

# Criar as restrições de oferta
if oferta_total > demanda_total:
    c1 = m.addConstrs(
        gp.quicksum(x[i, j] for j in clientes) <= ofertas[i] for i in fabricas)
else:
    c1 = m.addConstrs(
        gp.quicksum(x[i, j] for j in clientes) == ofertas[i] for i in fabricas)

# Criar as restrições de demanda
if oferta_total > demanda_total:
    c2 = m.addConstrs(
        gp.quicksum(x[i, j] for i in fabricas) == demandas[j] for j in clientes)
else:
    c2 = m.addConstrs(
        gp.quicksum(x[i, j] for i in fabricas) <= demandas[j] for j in clientes)

# Otimizar o modelo
m.optimize()

#Imprime o plano de transporte 
for i in fabricas:
    print("Origem:", i)
    for j in clientes:
        qtd = round(x[i, j].X)
        if qtd > 0:
            print("Trasnportar {} unidades para cliente {}".format(qtd, j))
    print("")

#Relatorio de oferta ou demanda desbalanceada
if oferta_total > demanda_total:
    print("As fabricas a seguir tem capacidade excdente:")
    for i in fabricas:
        sobra = round(c1[i].Slack)
        if sobra > 0:
            print("Fabrica {} tem {} unidades excedentes".format(i, sobra))

elif demanda_total > oferta_total:
    print("Os clientes a seguir tem demanda não atendida:")
    for j in clientes:
        falta = round(c2[j].Slack)
        if falta > 0:
            print("Cliente {} tem {} unidades de demanda não atendida".format(j, falta))

else:
    print("Oferta e demanda estão balanceadas")