import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

path_original_data = "./raw/leandl_oesnpg_dados.parquet"
df_original = pd.read_parquet(path_original_data)

# =====================================================================
# 1. LIMPEZA INICIAL
# =====================================================================
colunas_essenciais = [
    "hash_id", "tema_id", "modelo_nivel", "modelo_explicacao", "tema", "uf_tema_info", 
    "palavras_chave", "nome_producao", "descricao_resumo", 
    "descricao_palavra_chave", "nome_area_concentracao"
]

df_limpo = df_original.dropna(subset=colunas_essenciais).copy()

# =====================================================================
# 2. AGRUPAMENTO E CONCATENAÇÃO DE JUSTIFICATIVAS
# =====================================================================
df_comparison = (
    df_limpo[["hash_id", "tema_id", "modelo_nivel", "modelo_explicacao"]]
    .groupby(["hash_id", "tema_id"])
    .agg({
        "modelo_nivel": list,
        "modelo_explicacao": list,
    })
    .reset_index()
)

def processar_justificativa(lista_explicacoes):
    explicacoes_unicas = list(set([x for x in lista_explicacoes if x]))
    if not explicacoes_unicas:
        return "Justificativa não disponível."
    return ". ".join(explicacoes_unicas)

df_comparison["justificativa_final"] = df_comparison["modelo_explicacao"].apply(processar_justificativa)
df_comparison = df_comparison.drop(columns=["modelo_explicacao"])

# =====================================================================
# 3. FILTRAGEM DE CONFLITO E BINARIZAÇÃO DAS CLASSES
# =====================================================================

# Regra 1: Descarta casos extremos (ALTA e BAIXA simultaneamente)
def has_extreme_conflict(level_list):
    return {'ALTA', 'BAIXA'}.issubset(set(level_list))

mask_conflito = df_comparison["modelo_nivel"].apply(has_extreme_conflict)
df_comparison = df_comparison[~mask_conflito].copy()

# Regra 2 e 3: Classificação Binária (ALTA vs NAO-ALTA)
def level_filter_binario(level_list: list) -> str:
    niveis = set(level_list)
    # Como já removemos os conflitos de ALTA e BAIXA, se existir ALTA, a classe vence.
    if 'ALTA' in niveis:
        return 'ALTA'
    # Todo o resto (MEDIA/MEDIA, BAIXA/BAIXA, MEDIA/BAIXA) cai aqui.
    return 'NAO-ALTA'

df_comparison["nivel_classificacao"] = df_comparison["modelo_nivel"].apply(level_filter_binario)

# =====================================================================
# 4. PREPARAÇÃO DAS TABELAS TEMAS E PERFIS
# =====================================================================
def to_fset(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return frozenset(x)
    if pd.isna(x):
        return frozenset()
    return frozenset([x])

df_temas = (
    df_limpo.assign(palavras_chave=df_limpo["palavras_chave"].apply(to_fset))[
        ["tema_id", "tema", "uf_tema_info", "palavras_chave"]
    ]
    .drop_duplicates(subset=["tema_id"])
    .assign(
        palavras_chave=lambda df: (
            df["palavras_chave"].apply(list).apply(lambda kwords: ", ".join(kwords))
        )
    )
)

df_perfis = df_limpo.drop_duplicates(subset=["hash_id"])[
    [
        "hash_id", "nome_producao", "descricao_resumo", 
        "descricao_palavra_chave", "nome_area_concentracao"
    ]
].copy()

# =====================================================================
# 5. TRUNCAGEM DE ACORDO COM TOKENS
# =====================================================================
# Truncamento do texto para limitar max_seq_lenght em 2048 tokens, melhorando processamento
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-14B-unsloth-bnb-4bit")

# FUNÇÃO DE TRUNCAGEM DE RESUMO: Limita estritamente a 800 tokens.
# O restante do limite de 2048 acomodará o template do prompt e outras colunas.
def truncar_texto_por_tokens(texto, max_tokens=800):
    if pd.isna(texto) or texto == "":
        return texto
    
    # Tokeniza sem adicionar tokens especiais para medir o conteúdo real
    tokens = tokenizer.encode(texto, add_special_tokens=False)
    
    if len(tokens) > max_tokens:
        # Decodifica de volta para texto apenas os primeiros 'max_tokens'
        texto_truncado = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        return texto_truncado
    return texto

print("Aplicando truncagem segura nos resumos dos perfis...")
df_perfis["descricao_resumo"] = df_perfis["descricao_resumo"].apply(truncar_texto_por_tokens)

# =====================================================================
# 6. MERGE FINAL
# =====================================================================
print("Filtrando e Unindo Dataframes...")

df_final = df_comparison.merge(df_temas, on="tema_id", how="inner")
df_final = df_final.merge(df_perfis, on="hash_id", how="inner")

df_final = df_final[
    [
        "hash_id", "tema_id", "modelo_nivel", "justificativa_final", "nivel_classificacao",
        "tema", "uf_tema_info", "palavras_chave", "nome_producao",
        "descricao_resumo", "descricao_palavra_chave", "nome_area_concentracao",
    ]
]

df_final = df_final.rename(
    columns={
        "tema": "nome_tema",
        "uf_tema_info": "uf_tema",
        "palavras_chave": "palavras_chave_tema",
        "nome_producao": "nome_producao_perfil",
        "descricao_resumo": "resumo_perfil",
        "descricao_palavra_chave": "palavras_chave_perfil",
        "nome_area_concentracao": "area_concentracao_perfil",
    }
)

# =====================================================================
# 7. DIVISÃO ESTRATIFICADA (70/10/20) E EXPORTAÇÃO
# =====================================================================
print("Realizando split de 70/10/20...")
random_state = 133

# Corte 1: Isola 20% para Teste (sobram 80%)
df_temp, df_teste = train_test_split(
    df_final,
    test_size=0.20,
    random_state=random_state,
    stratify=df_final["uf_tema"]
)

# Corte 2: Separa os 80% restantes em 70% Treino e 10% Validação
# 10% do total equivale a 12.5% (0.125) da fatia de 80% restante
df_treino, df_validacao = train_test_split(
    df_temp,
    test_size=0.125, 
    random_state=random_state,
    stratify=df_temp["uf_tema"]
)

print(f"Total: {len(df_final)}")
print(f"Treino: {len(df_treino)} | Validação: {len(df_validacao)} | Teste: {len(df_teste)}")

df_treino.to_parquet("../data/processed/2class/df_treino_estratificado_70_10_20_2class_truncated.parquet")
df_validacao.to_parquet("../data/processed/2class/df_validacao_estratificado_70_10_20_2class_truncated.parquet")
df_teste.to_parquet("../data/processed/2class/df_teste_estratificado_70_10_20_2class_truncated.parquet")
