import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_final = pd.read_parquet("../data/processed/3class/df_completo_3class_truncated.parquet")

# Até aqui igual a filtragem de 3 classes

# === REMOÇÃO DA CLASSE 'MEDIA' PARA ABORDAGEM DE 2 CLASSES ===
print("Removendo classe MEDIA para manter apenas ALTA e BAIXA...")
df_final = df_final[df_final["nivel_classificacao"].isin(["ALTA", "BAIXA"])].reset_index(drop=True)

# =====================================================================
# 1. DIVISÃO DO DATASET (ESTRATIFICADO)
# =====================================================================
random_state = 133

# PRIMEIRO CORTE: Separa o Teste (20%) do Restante (80%)
df_temp, df_teste = train_test_split(
    df_final,
    test_size=0.20,
    random_state=random_state,
    stratify=df_final["uf_tema"]
)

# SEGUNDO CORTE: Divide o Restante em Treino (70%) e Validação (10%)
# A proporção de validação aqui é 10/80 (0.125) do dataset total
df_treino, df_validacao = train_test_split(
    df_temp,
    test_size=0.125, 
    random_state=random_state,
    stratify=df_temp["uf_tema"]
)

# =====================================================================
# 2. VERIFICAÇÃO E EXPORTAÇÃO
# =====================================================================
total = len(df_final)
print(f"Total original (pós-filtro 2 classes): {total} linhas")
print(f"Treino (70%): {len(df_treino)} linhas ({len(df_treino)/total:.1%})")
print(f"Validação (10%): {len(df_validacao)} linhas ({len(df_validacao)/total:.1%})")
print(f"Teste (20%): {len(df_teste)} linhas ({len(df_teste)/total:.1%})")

# Salvando df_final
df_final.to_parquet("../data/processed/2class/df_completo_2class_truncated.parquet")

# Salvando subsets
df_treino.to_parquet("../data/processed/2class/df_treino_estratificado_70_10_20_2class_truncated.parquet")
df_validacao.to_parquet("../data/processed/2class/df_validacao_estratificado_70_10_20_2class_truncated.parquet")
df_teste.to_parquet("../data/processed/2class/df_teste_estratificado_70_10_20_2class_truncated.parquet")

print("Exportação finalizada com sucesso!")