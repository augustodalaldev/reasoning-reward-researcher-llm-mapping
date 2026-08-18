import pandas as pd
import os
import glob

# Define as pastas de entrada e saída
pasta_entrada = "results"
pasta_saida = "results_default_labeled"

# Cria a pasta de saída se ela ainda não existir
os.makedirs(pasta_saida, exist_ok=True)

# Encontra todos os arquivos .parquet dentro da pasta de entrada
arquivos_parquet = glob.glob(os.path.join(pasta_entrada, "*.parquet"))

if not arquivos_parquet:
    print(f"Nenhum arquivo .parquet foi encontrado na pasta '{pasta_entrada}'.")
else:
    for caminho_arquivo in arquivos_parquet:
        # Extrai apenas o nome do arquivo (ex: 'dados.parquet')
        nome_arquivo = os.path.basename(caminho_arquivo)
        
        # Carrega o arquivo parquet em um DataFrame
        df = pd.read_parquet(caminho_arquivo)
        
        # Verifica se a coluna 'predicted' existe para evitar erros
        if 'predicted' in df.columns:
            # Filtra onde a coluna é extamente igual a "" e substitui por "BAIXA"
            df.loc[df['predicted'] == "", 'predicted'] = "BAIXA"
        else:
            print(f"Atenção: A coluna 'predicted' não existe no arquivo {nome_arquivo}.")
        
        # Define o caminho de salvamento do novo arquivo
        caminho_salvamento = os.path.join(pasta_saida, nome_arquivo)
        
        # Salva o arquivo atualizado no novo diretório
        df.to_parquet(caminho_salvamento)
        
    print(f"Processamento concluído! {len(arquivos_parquet)} arquivo(s) salvo(s) em '{pasta_saida}'.")