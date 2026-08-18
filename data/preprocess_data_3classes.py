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

# Removemos qualquer linha que tenha NaN nas colunas que usaremos
df_limpo = df_original.dropna(subset=colunas_essenciais).copy()

# =====================================================================
# 2. REMOÇÃO DE CONFLITOS EXTREMOS 
# =====================================================================
conflitos = df_limpo.groupby(['hash_id', 'tema_id'])['modelo_nivel'].apply(set)
pares_extremos = conflitos[conflitos.apply(lambda x: {'ALTA', 'BAIXA'}.issubset(x))].index

df_limpo = df_limpo.set_index(['hash_id', 'tema_id'])
df_limpo = df_limpo[~df_limpo.index.isin(pares_extremos)].reset_index()

# =====================================================================
# 3. RESOLUÇÃO DE CONFLITOS
# Regra: ALTA > BAIXA > MEDIA
# =====================================================================
ordem_prioridade = {'ALTA': 1, 'BAIXA': 2, 'MEDIA': 3}
df_limpo['prioridade'] = df_limpo['modelo_nivel'].map(ordem_prioridade)

# Ordenamos pelo ID, Tema e nossa Prioridade
df_limpo = df_limpo.sort_values(by=['hash_id', 'tema_id', 'prioridade'])

# Ao remover as duplicatas mantendo a 'first' (primeira), resolvemos todas as suas regras:
# - Duas da mesma classe? Fica a primeira.
# - ALTA e MEDIA? ALTA é prioridade 1, fica no topo e é mantida.
# - BAIXA e MEDIA? BAIXA é prioridade 2, fica no topo e é mantida.
# O mais importante: a coluna 'modelo_explicacao' da linha vencedora é preservada intacta!
df_resolvido = df_limpo.drop_duplicates(subset=['hash_id', 'tema_id'], keep='first').copy()
df_resolvido = df_resolvido.drop(columns=['prioridade'])

# Criamos as colunas finais baseadas na resolução
df_resolvido["justificativa_final"] = df_resolvido["modelo_explicacao"]
df_resolvido["nivel_classificacao"] = df_resolvido["modelo_nivel"]

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
    df_resolvido.assign(palavras_chave=df_resolvido["palavras_chave"].apply(to_fset))[
        ["tema_id", "tema", "uf_tema_info", "palavras_chave"]
    ]
    .drop_duplicates(subset=["tema_id"])
    .assign(
        palavras_chave=lambda df: (
            df["palavras_chave"].apply(list).apply(lambda kwords: ", ".join(kwords))
        )
    )
)

# df só contendo os perfis
df_perfis = df_resolvido.drop_duplicates(subset=["hash_id"])[
    [
        "hash_id", "nome_producao", "descricao_resumo", 
        "descricao_palavra_chave", "nome_area_concentracao"
    ]
].copy()

# =====================================================================
# 6. TRUNCAGEM DE ACORDO COM TOKENS
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
# 5. MERGE FINAL E EXPORTAÇÃO
# =====================================================================
print("Filtrando e Unindo Dataframes...")

df_final = df_resolvido[['hash_id', 'tema_id', 'modelo_nivel', 'justificativa_final', 'nivel_classificacao']]
df_final = df_final.merge(df_temas, on="tema_id", how="inner")
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


# Divisão do dataset 

random_state = 133

# =====================================================================
# 1. PRIMEIRO CORTE: Separa o Teste (20%) do Restante (80%)
# =====================================================================
df_temp, df_teste = train_test_split(
    df_final,
    test_size=0.20,
    random_state=random_state,
    stratify=df_final["uf_tema"]
)

# =====================================================================
# 2. SEGUNDO CORTE: Divide o Restante em Treino (70%) e Validação (10%)
# A proporção de teste aqui é 10/80 (0.125)
# =====================================================================
df_treino, df_validacao = train_test_split(
    df_temp,
    test_size=0.125, 
    random_state=random_state,
    stratify=df_temp["uf_tema"]
)

# =====================================================================
# 3. VERIFICAÇÃO E EXPORTAÇÃO
# =====================================================================
total = len(df_final)
print(f"Total original: {total} linhas")
print(f"Treino (70%): {len(df_treino)} linhas ({len(df_treino)/total:.1%})")
print(f"Validação (10%): {len(df_validacao)} linhas ({len(df_validacao)/total:.1%})")
print(f"Teste (20%): {len(df_teste)} linhas ({len(df_teste)/total:.1%})")


# Salvando no diretório especificado

# Salvando df_final
df_final.to_parquet("../data/processed/3class/df_completo_3class_truncated.parquet")

# Salvando subsets
df_treino.to_parquet("../data/processed/3class/df_treino_estratificado_70_10_20_3class_truncated.parquet")
df_validacao.to_parquet("../data/processed/3class/df_validacao_estratificado_70_10_20_3class_truncated.parquet")
df_teste.to_parquet("../data/processed/3class/df_teste_estratificado_70_10_20_3class_truncated.parquet")

