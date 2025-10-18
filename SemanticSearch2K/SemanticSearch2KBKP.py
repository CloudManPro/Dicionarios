import os
import time
import json
import zipfile
import boto3
import pickle
import numpy as np
import re
import urllib.request
import urllib.error
# REMOVIDO: import google.generativeai as genai
import traceback # Para logs de erro detalhados

# ----------------- Configurações -----------------
S3_BUCKET = os.getenv("AWS_S3_BUCKET_TARGET_NAME_0")
INDEX_ZIP_KEY = "Semantic2K.zip"    # Arquivo zip com o índice
TMP_INDEX_DIR = "/tmp/index"        # Diretório temporário
INDEX_FILE = os.path.join(TMP_INDEX_DIR, "index.pkl")  # Arquivo pickle do índice

# --- Configuração da API do Google GenAI (APENAS REST) ---

API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCdOsmdzquKHolKf6WdWhFNkbRMAfkdF_E") # Use environment variable

# --- Modelos ---
MODEL_NAME_EMBEDDING = "text-embedding-004"        # Modelo para gerar embeddings
MODEL_NAME_COUNTING = "gemini-1.5-flash-latest"    # Modelo GENERATIVO para contar tokens (precisa suportar countTokens)

# --- Nomes completos para API REST ---
MODEL_NAME_EMBEDDING_API = f"models/{MODEL_NAME_EMBEDDING}"
MODEL_NAME_COUNTING_API = f"models/{MODEL_NAME_COUNTING}" # Usado na URL de contagem

# --- Endpoints REST ---
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_URL_EMBED_CONTENT = f"{API_BASE_URL}/{MODEL_NAME_EMBEDDING_API}:embedContent?key={API_KEY}"
API_URL_BATCH_EMBED = f"{API_BASE_URL}/{MODEL_NAME_EMBEDDING_API}:batchEmbedContents?key={API_KEY}"
API_URL_COUNT_TOKENS = f"{API_BASE_URL}/{MODEL_NAME_COUNTING_API}:countTokens?key={API_KEY}" # <--- NOVO ENDPOINT

MAX_TOKENS_PER_CHUNK = 2040 # Limite seguro para text-embedding-004 (chunking logic relies on token count)
TOKENS_SEPARADOR_ESTIMADO = 2 # Estimativa para '\n\n' ao somar localmente

print(f"DEBUG: S3_BUCKET = {S3_BUCKET}")
print(f"DEBUG: INDEX_ZIP_KEY = {INDEX_ZIP_KEY}")
print(f"DEBUG: MODEL_NAME_EMBEDDING = {MODEL_NAME_EMBEDDING}")
print(f"DEBUG: MODEL_NAME_COUNTING = {MODEL_NAME_COUNTING}")
print(f"DEBUG: MODEL_NAME_EMBEDDING_API = {MODEL_NAME_EMBEDDING_API}")
print(f"DEBUG: MODEL_NAME_COUNTING_API = {MODEL_NAME_COUNTING_API}")
print(f"DEBUG: API_URL_EMBED_CONTENT = {API_URL_EMBED_CONTENT}")
print(f"DEBUG: API_URL_BATCH_EMBED = {API_URL_BATCH_EMBED}")
print(f"DEBUG: API_URL_COUNT_TOKENS = {API_URL_COUNT_TOKENS}") # <--- Log do novo endpoint
print(f"DEBUG: MAX_TOKENS_PER_CHUNK = {MAX_TOKENS_PER_CHUNK}")

# Inicializa o cliente S3
s3 = boto3.client('s3')

# Índice global
index = {}

# REMOVIDO: gemini_model_for_counting = None

# ----------------------------------------------------------------------------

# ----------------- Funções Auxiliares -----------------
def strip_html_tags(text):
    """Remove tags HTML de uma string."""
    if not isinstance(text, str): return "" # Lida com entrada não-string
    return re.sub('<[^<]+?>', '', text)

# ----- *** NOVA Função count_tokens_func_rest *** -----
def count_tokens_func_rest(text_to_count):
    """Conta tokens usando a API REST do Google GenAI."""
    if not API_KEY:
        print("ERRO: API Key não configurada, não é possível contar tokens via REST.")
        return float('inf')
    if not text_to_count or not isinstance(text_to_count, str) or not text_to_count.strip():
         return 0 # Texto vazio ou inválido tem 0 tokens

    # Payload para a API countTokens (geralmente precisa do 'contents')
    # A estrutura pode variar ligeiramente, mas esta é comum para modelos Gemini
    payload = {
        # "model": MODEL_NAME_COUNTING_API, # Opcional se já estiver na URL, mas pode ser necessário
        "contents": [{"parts": [{"text": text_to_count}]}]
    }
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_URL_COUNT_TOKENS, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                error_body = response.read().decode('utf-8', errors='ignore')
                # Verifica se o erro é específico de modelo não suportado
                if "is not found for API version v1beta, or is not supported for countTokens" in error_body:
                     print(f"ERRO CRÍTICO (REST): Modelo '{MODEL_NAME_COUNTING}' ({MODEL_NAME_COUNTING_API}) não suporta countTokens via API REST. Verifique o nome do modelo.")
                else:
                     print(f"ERRO API {response.status} (countTokens REST): {error_body}")
                return float('inf') # Indica erro na contagem

            response_data = response.read().decode("utf-8")
        result = json.loads(response_data)

        if "totalTokens" not in result:
             print(f"AVISO: Resposta inesperada da API countTokens (REST), 'totalTokens' não encontrado: {result}")
             return float('inf') # Indica erro

        return int(result["totalTokens"])

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        print(f"DEBUG: Erro HTTP {e.code} em count_tokens_func_rest: {e.reason}")
        print(f"DEBUG: Resposta da API (se houver): {error_body}")
        # Verifica novamente o erro específico do modelo
        if e.code == 404 and "is not found for API version v1beta, or is not supported for countTokens" in error_body:
             print(f"ERRO CRÍTICO (REST HTTP 404): Modelo '{MODEL_NAME_COUNTING}' ({MODEL_NAME_COUNTING_API}) não suporta countTokens via API REST.")
        return float('inf') # Indica erro
    except Exception as e:
        print(f"ERRO: Erro genérico em count_tokens_func_rest para texto '{text_to_count[:100]}...': {str(e)}")
        print(traceback.format_exc())
        return float('inf') # Indica erro
# -----------------------------------------------------

# ----- Função split_text_into_chunks_by_tokens -----
# Modificada para usar count_tokens_func_rest
def split_text_into_chunks_by_tokens(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    """
    Divide o texto em blocos baseados em parágrafos, respeitando o limite de tokens.
    Utiliza a API REST para count_tokens.
    """
    print(f"DEBUG: Iniciando split_text_into_chunks_by_tokens (REST-based count) com max_tokens={max_tokens}")
    final_chunks = []
    if not text or not text.strip():
        print("DEBUG: Texto de entrada vazio ou inválido.")
        return []

    # 1. Divide em parágrafos (igual a antes)
    parts = re.split(r'(\n{2,})', text)
    paragraphs = []
    current_paragraph = ""
    for i, part in enumerate(parts):
        if i % 2 == 0: current_paragraph += part
        else:
            current_paragraph += part
            if current_paragraph.strip(): paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
    if current_paragraph.strip(): paragraphs.append(current_paragraph.strip())

    if not paragraphs:
        print("DEBUG: Não foram encontrados separadores de parágrafo (\\n\\n+). Tratando como um único bloco.")
        paragraphs = [text.strip()]

    print(f"DEBUG: Texto dividido em {len(paragraphs)} parágrafos iniciais.")

    # 2. Pré-calcula tokens para cada parágrafo (Usando REST API agora)
    paragraphs_with_tokens = []
    print(f"DEBUG: Iniciando contagem de tokens por parágrafo via REST API (modelo: {MODEL_NAME_COUNTING})...") # Added model name
    start_count_time = time.time()
    for i, p in enumerate(paragraphs):
        token_count = count_tokens_func_rest(p) # <--- CHAMADA DA API REST AQUI
        if token_count == float('inf'):
            print(f"ERRO: Falha ao contar tokens para o parágrafo {i} via REST API. Parágrafo será ignorado.")
            continue
        # print(f"DEBUG: Parágrafo {i} tem {token_count} tokens (via REST).") # Log pode ficar verboso
        paragraphs_with_tokens.append({'text': p, 'tokens': token_count})
        if token_count > max_tokens:
            print(f"!!! ALERTA: Parágrafo {i} ({token_count} tokens) excede o limite de {max_tokens} sozinho!")
    end_count_time = time.time()
    print(f"DEBUG: Contagem de tokens para {len(paragraphs_with_tokens)} parágrafos concluída em {end_count_time - start_count_time:.2f} seg (via REST).")

    # 3. Constrói os Chunks iterativamente (igual a antes, usa contagens pré-calculadas)
    current_chunk_paragraphs = []
    current_chunk_tokens_estimated = 0

    for item in paragraphs_with_tokens:
        paragraph_text = item['text']
        paragraph_tokens = item['tokens']

        if paragraph_tokens > max_tokens:
            if current_chunk_paragraphs:
                chunk_text = "\n\n".join(current_chunk_paragraphs)
                final_chunks.append(chunk_text)
                print(f"DEBUG: Chunk finalizado (antes do parágrafo grande).")
                current_chunk_paragraphs = []
                current_chunk_tokens_estimated = 0
            final_chunks.append(paragraph_text)
            print(f"DEBUG: Parágrafo grande ({paragraph_tokens} tokens) adicionado como chunk separado.")
            continue

        tokens_separador_a_adicionar = 0 if not current_chunk_paragraphs else TOKENS_SEPARADOR_ESTIMADO
        potential_tokens = current_chunk_tokens_estimated + tokens_separador_a_adicionar + paragraph_tokens

        if potential_tokens <= max_tokens:
            current_chunk_paragraphs.append(paragraph_text)
            current_chunk_tokens_estimated = potential_tokens
        else:
            if current_chunk_paragraphs:
                chunk_text = "\n\n".join(current_chunk_paragraphs)
                final_chunks.append(chunk_text)
            current_chunk_paragraphs = [paragraph_text]
            current_chunk_tokens_estimated = paragraph_tokens

    if current_chunk_paragraphs:
        chunk_text = "\n\n".join(current_chunk_paragraphs)
        final_chunks.append(chunk_text)

    # 4. Log Final e Verificação (Usando REST API novamente)
    verified_chunks = []
    print("-" * 20)
    print("DEBUG: Verificando tokens dos chunks finais via REST API...")
    start_verify_time = time.time()
    for i, chunk in enumerate(final_chunks):
        final_token_count = count_tokens_func_rest(chunk) # <--- CHAMADA DA API REST AQUI
        print(f"DEBUG: Chunk {i+1} - Tokens: {final_token_count} (via REST) - Texto (primeiros 100): '{chunk[:100]}...'")
        if final_token_count > MAX_TOKENS_PER_CHUNK:
             print(f"!!! ALERTA FINAL: Chunk {i+1} ({final_token_count} tokens) ainda excede o limite {MAX_TOKENS_PER_CHUNK}!")
        if final_token_count == 0 or final_token_count == float('inf'):
            print(f"AVISO: Chunk {i+1} tem contagem de tokens 0 ou infinita (REST). Será ignorado.")
        else:
            verified_chunks.append(chunk)
    end_verify_time = time.time()
    print(f"DEBUG: Verificação de {len(final_chunks)} chunks concluída em {end_verify_time - start_verify_time:.2f} seg (via REST).")
    print("-" * 20)

    print(f"DEBUG: Retornando {len(verified_chunks)} chunks válidos para embedding.")
    return verified_chunks
# ----------------------------------------------

# --- Funções de Embedding (USANDO REST API - sem alterações) ---
def get_embedding(text):
    # ... (código igual ao anterior)
    if not API_KEY: raise ValueError("API Key não configurada para embedding.")
    if not text or not text.strip():
         raise ValueError("Texto vazio fornecido para get_embedding.")

    payload = {"model": MODEL_NAME_EMBEDDING_API, "content": {"parts": [{"text": text}]}}
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_URL_EMBED_CONTENT, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                 error_body = response.read().decode('utf-8', errors='ignore')
                 raise Exception(f"API Error {response.status} (embedContent): {error_body}")
            response_data = response.read().decode("utf-8")
        result = json.loads(response_data)
        if "embedding" not in result or "values" not in result.get("embedding", {}):
             raise Exception(f"Resposta inesperada da API (embedding não encontrado): {result}")
        embedding_list = result["embedding"]["values"]
        embedding = np.array(embedding_list, dtype=np.float32).flatten()
        return embedding
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        print(f"DEBUG: Erro HTTP {e.code} em get_embedding (REST): {e.reason}")
        print(f"DEBUG: Resposta da API (se houver): {error_body}")
        raise Exception(f"Erro HTTP {e.code} ao chamar API de embedding (REST).") from e
    except Exception as e:
        print(f"DEBUG: Erro genérico em get_embedding (REST) para texto '{text[:100]}...': {str(e)}")
        print(traceback.format_exc())
        raise

def get_embeddings(texts):
    # ... (código igual ao anterior)
    if not API_KEY: raise ValueError("API Key não configurada para batch embedding.")
    valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
    if not valid_texts:
        print("DEBUG: Nenhum texto válido fornecido para get_embeddings em lote.")
        return []

    payload = {
        "requests": [
            {"model": MODEL_NAME_EMBEDDING_API, "content": {"parts": [{"text": t}]}}
            for t in valid_texts
        ]
    }
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_URL_BATCH_EMBED, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                 error_body = response.read().decode('utf-8', errors='ignore')
                 raise Exception(f"API Error {response.status} (batchEmbed): {error_body}")
            response_data = response.read().decode("utf-8")
        result = json.loads(response_data)
        if "embeddings" not in result:
            raise Exception(f"Resposta inesperada da API (batch embeddings não encontrado): {result}")

        if len(result["embeddings"]) != len(valid_texts):
             print(f"AVISO: Discrepância no batch. Requests: {len(valid_texts)}, Embeddings: {len(result['embeddings'])}. Resposta: {result}")
             raise Exception(f"Discrepância entre número de requests ({len(valid_texts)}) e embeddings recebidos ({len(result['embeddings'])}).")

        embeddings = []
        for i, embedding_data in enumerate(result["embeddings"]):
            if "values" not in embedding_data:
                print(f"AVISO: Embedding sem 'values' encontrado na resposta batch para request {i} (texto: '{valid_texts[i][:50]}...'). Resposta: {embedding_data}")
                raise Exception(f"Embedding inválido (sem 'values') na resposta batch para request {i}.")
            emb_list = embedding_data["values"]
            emb = np.array(emb_list, dtype=np.float32).flatten()
            embeddings.append(emb)

        print(f"DEBUG: Embeddings em lote geradas (REST) - count={len(embeddings)}")
        return embeddings
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        print(f"DEBUG: Erro HTTP {e.code} em get_embeddings (REST): {e.reason}")
        print(f"DEBUG: Resposta da API (se houver): {error_body}")
        raise Exception(f"Erro HTTP {e.code} ao chamar API de batch embedding (REST).") from e
    except Exception as e:
        print(f"DEBUG: Erro genérico em get_embeddings (REST): {str(e)}")
        print(traceback.format_exc())
        raise


# ----------------- Gerenciamento do índice (sem alterações) -----------------
def load_index():
    # ... (código igual ao anterior)
    print("DEBUG: Iniciando load_index()")
    global index # Ensure we modify the global index

    if not os.path.exists(TMP_INDEX_DIR):
        try:
            os.makedirs(TMP_INDEX_DIR)
            print(f"DEBUG: Diretório {TMP_INDEX_DIR} criado.")
        except OSError as e:
            print(f"ERRO: Falha ao criar diretório {TMP_INDEX_DIR}: {e}")
            return

    zip_path = "/tmp/index_download.zip"
    try:
        print(f"DEBUG: Baixando {INDEX_ZIP_KEY} de {S3_BUCKET} para {zip_path}")
        s3.download_file(S3_BUCKET, INDEX_ZIP_KEY, zip_path)
        print(f"DEBUG: Extraindo {zip_path} para {TMP_INDEX_DIR}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TMP_INDEX_DIR)
        print("DEBUG: Índice extraído do zip do S3.")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"DEBUG: Arquivo {INDEX_ZIP_KEY} não encontrado no bucket {S3_BUCKET} (provavelmente primeira execução).")
        else:
            print(f"ERRO S3 ClientError ao baixar/extrair índice: {e}")
    except FileNotFoundError:
         print(f"DEBUG: Arquivo zip {zip_path} não encontrado após tentativa de download.")
    except zipfile.BadZipFile:
         print(f"ERRO: Arquivo {zip_path} baixado parece ser um zip inválido.")
    except Exception as e:
        print(f"ERRO inesperado ao baixar ou extrair índice do S3: {e}")
        print(traceback.format_exc())


    if os.path.exists(INDEX_FILE):
        print(f"DEBUG: Encontrado arquivo {INDEX_FILE}. Carregando...")
        try:
            with open(INDEX_FILE, "rb") as f:
                index_loaded = pickle.load(f)
            processed_index = {}
            for doc_id, doc in index_loaded.items():
                 if "embedding" in doc and isinstance(doc["embedding"], list):
                     doc["embedding"] = np.array(doc["embedding"], dtype=np.float32).flatten()
                 elif doc.get("embedding") is None or not isinstance(doc.get("embedding"), np.ndarray):
                      doc["embedding"] = None
                 if "parts" in doc and isinstance(doc["parts"], list):
                     for i, part in enumerate(doc["parts"]):
                         if "embedding" in part and isinstance(part["embedding"], list):
                             part["embedding"] = np.array(part["embedding"], dtype=np.float32)
                         elif part.get("embedding") is None or not isinstance(part.get("embedding"), np.ndarray):
                              part["embedding"] = None
                 elif "parts" in doc: print(f"AVISO: 'parts' no documento {doc_id} não é uma lista.")
                 processed_index[doc_id] = doc

            index = processed_index
            print(f"DEBUG: Índice carregado e processado de {INDEX_FILE}. Total de documentos: {len(index)}")
        except EOFError:
             print(f"ERRO: Arquivo pickle {INDEX_FILE} parece estar vazio ou corrompido (EOFError). Resetando índice.")
             index = {}
        except pickle.UnpicklingError as e:
             print(f"ERRO: Falha ao deserializar o arquivo pickle {INDEX_FILE}: {e}. Resetando índice.")
             index = {}
        except Exception as e:
            print(f"ERRO: Falha genérica ao carregar ou processar o arquivo pickle {INDEX_FILE}: {e}")
            print(traceback.format_exc())
            index = {}
    else:
        index = {}
        print(f"DEBUG: {INDEX_FILE} não existe. Criando índice vazio.")


def save_index():
    # ... (código igual ao anterior)
    print(f"DEBUG: save_index() - Tamanho do índice atual: {len(index)}")
    if not os.path.exists(TMP_INDEX_DIR):
         print(f"AVISO: Diretório {TMP_INDEX_DIR} não existe. Tentando criar para salvar o índice.")
         try:
             os.makedirs(TMP_INDEX_DIR)
         except Exception as e:
             print(f"ERRO: Não foi possível criar {TMP_INDEX_DIR} para salvar o índice. Abortando save_index(). Erro: {e}")
             return

    index_to_save = {}
    for doc_id, doc_data in index.items():
        saved_doc = {}
        for key, value in doc_data.items():
             if key == "embedding" and isinstance(value, np.ndarray):
                 saved_doc[key] = value.tolist()
             elif key == "parts" and isinstance(value, list):
                 saved_parts = []
                 for part in value:
                     saved_part = part.copy()
                     if isinstance(saved_part.get("embedding"), np.ndarray):
                         saved_part["embedding"] = saved_part["embedding"].tolist()
                     saved_parts.append(saved_part)
                 saved_doc[key] = saved_parts
             else:
                 saved_doc[key] = value
        index_to_save[doc_id] = saved_doc


    try:
        with open(INDEX_FILE, "wb") as f:
            pickle.dump(index_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"DEBUG: Índice (com embeddings como listas) salvo em {INDEX_FILE}")
    except Exception as e:
        print(f"ERRO: Falha ao salvar o índice em {INDEX_FILE}: {e}")
        print(traceback.format_exc())
        return

    zip_path = "/tmp/index_upload.zip"
    try:
        print(f"DEBUG: Criando arquivo zip {zip_path} contendo {os.path.basename(INDEX_FILE)}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(INDEX_FILE):
                 zipf.write(INDEX_FILE, os.path.basename(INDEX_FILE))
            else:
                 print(f"ERRO: {INDEX_FILE} não encontrado após salvar para adicionar ao zip.")
                 return

        print(f"DEBUG: Enviando {zip_path} para S3 bucket={S3_BUCKET}, key={INDEX_ZIP_KEY}")
        s3.upload_file(zip_path, S3_BUCKET, INDEX_ZIP_KEY)
        print("DEBUG: Índice atualizado enviado ao S3.")
    except FileNotFoundError:
         print(f"ERRO: Não foi possível encontrar {zip_path} para fazer upload.")
    except Exception as e:
        print(f"ERRO: Falha ao criar zip ou fazer upload para o S3: {e}")
        print(traceback.format_exc())


def delete_document(doc_id):
    # ... (código igual ao anterior)
    global index
    print(f"DEBUG: delete_document() - doc_id={doc_id}")
    if doc_id in index:
        del index[doc_id]
        print(f"DEBUG: Documento {doc_id} removido do índice em memória.")
        try:
            save_index()
            return {"status": "sucesso", "message": f"Documento {doc_id} removido com sucesso."}
        except Exception as e:
            print(f"ERRO: Documento {doc_id} removido da memória, MAS FALHA AO SALVAR ÍNDICE: {e}")
            return {"status": "error", "message": f"Documento {doc_id} removido da memória, mas erro ao persistir a remoção."}
    else:
        print(f"DEBUG: Documento {doc_id} não encontrado no índice.")
        return {"status": "error", "message": f"Documento {doc_id} não encontrado."}


# ----------------- Função de Similaridade (sem alterações) -----------------
def cosine_similarity(vec1, vec2):
    # ... (código igual ao anterior)
    if vec1 is None or vec2 is None or not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray): return 0.0
    try: vec1 = vec1.flatten(); vec2 = vec2.flatten()
    except AttributeError: print(f"DEBUG: Falha ao achatar vetores para similaridade. Tipos: {type(vec1)}, {type(vec2)}"); return 0.0
    if vec1.shape[0] == 0 or vec2.shape[0] == 0: return 0.0
    if vec1.shape != vec2.shape: print(f"AVISO: Formatos de vetor incompatíveis para similaridade: {vec1.shape} vs {vec2.shape}"); return 0.0
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0: return 0.0
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    return float(np.clip(similarity, -1.0, 1.0))

# ----------------- Função de Busca (sem alterações, já usa REST para query embedding) -----------------
def search_documents(keywords, match_type="or", max_results=10, full=False):
    # ... (código igual ao anterior)
    print(f"DEBUG: search_documents() - keywords={keywords}, match_type={match_type}, max_results={max_results}, full={full}")
    if not API_KEY: print("ERRO: API Key não configurada, não é possível gerar embedding da consulta."); return []
    if not keywords or not isinstance(keywords, list) or not any(k.strip() for k in keywords if isinstance(k, str)): print("DEBUG: keywords inválida ou vazia."); return []
    query_text = " ".join(k for k in keywords if isinstance(k, str) and k.strip())
    try: query_embedding = get_embedding(query_text) # Usa REST
    except Exception as e: print(f"ERRO: Falha ao gerar embedding da consulta '{query_text}' via REST: {e}"); return []
    if query_embedding is None: print("ERRO: Embedding da consulta retornou None."); return []
    results_list = []; all_snippets = []
    for doc_id, doc in index.items():
        doc_content = doc.get("content", ""); doc_content_lower = doc_content.lower() if isinstance(doc_content, str) else ""
        valid_keywords = [kw.lower() for kw in keywords if isinstance(kw, str) and kw.strip()]
        if match_type == "and":
            if not all(keyword in doc_content_lower for keyword in valid_keywords): continue
        if full:
            doc_embedding = doc.get("embedding")
            if doc_embedding is not None:
                doc_score = cosine_similarity(query_embedding, doc_embedding)
                results_list.append({"id": doc_id, "content": doc_content, "score": doc_score})
            else: print(f"AVISO: Documento {doc_id} sem embedding combinado, não pode ser incluído na busca 'full'.")
        else:
            doc_parts = doc.get("parts")
            if isinstance(doc_parts, list) and doc_parts:
                has_valid_parts = False
                for i, part in enumerate(doc_parts):
                    part_emb = part.get("embedding"); part_text = part.get("text", "")
                    if part_emb is not None and isinstance(part_text, str) and part_text.strip():
                        part_score = cosine_similarity(query_embedding, part_emb)
                        all_snippets.append({"doc_id": doc_id, "text": part_text, "score": part_score})
                        has_valid_parts = True
            else:
                doc_embedding = doc.get("embedding")
                if doc_embedding is not None and doc_content:
                     print(f"DEBUG: Doc {doc_id} sem 'parts' válidas, usando embedding combinado como fallback.")
                     doc_score = cosine_similarity(query_embedding, doc_embedding)
                     all_snippets.append({"doc_id": doc_id, "text": doc_content, "score": doc_score})
    if full:
        results_list.sort(key=lambda x: x["score"], reverse=True)
        print(f"DEBUG: Busca 'full' retornando {len(results_list[:max_results])} resultados.")
        return results_list[:max_results]
    else:
        all_snippets.sort(key=lambda x: x["score"], reverse=True)
        top_n_snippets = 30
        top_snippets = all_snippets[:top_n_snippets]
        grouped_results = {}
        for snippet in top_snippets:
            doc_id = snippet["doc_id"]
            if doc_id not in grouped_results:
                grouped_results[doc_id] = {"id": doc_id, "score": -1.0, "content": []}
            grouped_results[doc_id]["content"].append({"text": snippet["text"], "score": snippet["score"]})
            grouped_results[doc_id]["score"] = max(grouped_results[doc_id]["score"], snippet["score"])
        final_results_list = sorted(list(grouped_results.values()), key=lambda x: x["score"], reverse=True)
        print(f"DEBUG: Busca por snippets retornando {len(final_results_list[:max_results])} documentos agrupados.")
        return final_results_list[:max_results]


# ----------------- Função Lambda Handler -----------------
def lambda_handler(event, context):
    # REMOVIDO: Bloco de inicialização do gemini_model_for_counting

    start_time = time.time()
    print(f"DEBUG: Evento recebido: {json.dumps(event, indent=2)}")

    # Verifica se a API Key está disponível (essencial para TODAS as chamadas REST)
    if not API_KEY:
        print("ERRO FATAL: GOOGLE_API_KEY não configurada.")
        return {
             "statusCode": 500,
             "body": json.dumps({"status": "error", "message": "Erro interno: Configuração da API ausente."})
         }

    # Evento de aquecimento
    if event.get("source") == "aws.events" or event.get("WARMUP") == "TRUE":
        print("DEBUG: Evento de aquecimento recebido.")
        # Pode fazer um teste rápido de countTokens REST aqui se quiser
        try:
            test_count = count_tokens_func_rest("warmup test")
            if test_count != float('inf'):
                 print(f"DEBUG: Teste de contagem REST no aquecimento OK (tokens: {test_count}).")
            else:
                 print("AVISO: Teste de contagem REST no aquecimento FALHOU.")
        except Exception as e:
             print(f"AVISO: Exceção durante teste de contagem REST no aquecimento: {e}")
        return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "message": "Lambda aquecida."})}

    # Determina a ação
    action = event.get("action", "").lower()
    print(f"DEBUG: Ação solicitada: '{action}'")

    # ---- AÇÃO INSERT ----
    if action == "insert":
        doc_id = event.get("doc_id", f"doc_{int(time.time())}")
        print(f"DEBUG: Iniciando Ação INSERT para doc_id='{doc_id}'.")

        # Extração de texto (igual a antes)
        parts_list = event.get("parts")
        text_content = ""
        if isinstance(parts_list, list) and parts_list:
            potential_text = parts_list[0]
            if isinstance(potential_text, str): text_content = potential_text
        if not text_content: text_content = event.get("content", event.get("text", ""))
        if not text_content or not text_content.strip():
            print("ERRO: Conteúdo textual final está vazio.")
            return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "Conteúdo textual não encontrado ou vazio."})}

        # Limpeza HTML (igual a antes)
        cleaned_text = strip_html_tags(text_content)
        if not cleaned_text.strip():
             print("ERRO: Conteúdo textual ficou vazio após limpeza de HTML.")
             return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "Conteúdo vazio após limpeza."})}

        # ** CHUNKING (agora usa count_tokens_func_rest) **
        try:
            chunks = split_text_into_chunks_by_tokens(cleaned_text, MAX_TOKENS_PER_CHUNK)
        except Exception as e:
             print(f"ERRO: Falha durante o chunking (REST-based count): {e}")
             print(traceback.format_exc())
             # Retorna erro genérico, a falha específica da contagem já foi logada dentro da função
             return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno durante a divisão do texto: {e}"})}

        if not chunks:
            print("AVISO: Nenhum chunk válido gerado após divisão. Documento não será indexado.")
            return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "Nenhum chunk válido gerado a partir do texto."})}

        print(f"DEBUG: Texto dividido em {len(chunks)} chunks válidos para embedding.")

        # ** EMBEDDING (usa REST API - sem alterações) **
        chunk_embeddings = []
        combined_embedding = None
        try:
            chunk_embeddings = get_embeddings(chunks) # Chamada REST batch
            if len(chunk_embeddings) != len(chunks):
                 print(f"ERRO: Discrepância no batch embedding (REST). Esperado: {len(chunks)}, Recebido: {len(chunk_embeddings)}")
                 raise Exception("Falha ao obter embeddings para todos os chunks via REST.")
            if chunk_embeddings:
                 combined_embedding = np.mean(chunk_embeddings, axis=0).flatten()
                 print(f"DEBUG: Embedding combinado calculado (média dos chunks REST) - shape={combined_embedding.shape}")
            else:
                 print("AVISO: Nenhum embedding retornado pela API REST batch.")
                 combined_embedding = None
        except Exception as e:
            print(f"ERRO: Falha ao gerar embeddings (REST): {e}")
            print(traceback.format_exc())
            return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro ao gerar embeddings via REST: {str(e)}"}) }

        # Montagem da estrutura do índice (igual a antes)
        parts_info = []
        if len(chunks) == len(chunk_embeddings):
             parts_info = [{"text": part_text, "embedding": part_emb}
                           for part_text, part_emb in zip(chunks, chunk_embeddings)]
        else:
             print(f"ERRO: Despareamento final entre chunks ({len(chunks)}) e embeddings ({len(chunk_embeddings)}). Salvando sem 'parts'.")

        index[doc_id] = {
            "id": doc_id, "content": cleaned_text,
            "embedding": combined_embedding,
            "parts": parts_info if parts_info else []
        }
        print(f"DEBUG: Documento '{doc_id}' adicionado/atualizado no índice (embedding {'calculado' if combined_embedding is not None else 'NULO'}, {len(parts_info)} partes).")

        save_index() # Salva índice no S3

        end_time = time.time()
        print(f"DEBUG: Ação INSERT para '{doc_id}' concluída em {end_time - start_time:.2f} segundos.")
        return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "message": f"Documento {doc_id} indexado com sucesso ({len(chunks)} chunks)."})}

    # ---- AÇÃO DELETE (sem alterações) ----
    elif action == "delete":
        # ... (código igual ao anterior)
        doc_id = event.get("id", event.get("doc_id", ""))
        print(f"DEBUG: Ação DELETE para doc_id='{doc_id}'")
        if not doc_id: return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "ID (id/doc_id) não fornecido."})}
        result = delete_document(doc_id)
        status_code = 200 if result.get("status") == "sucesso" else 404
        end_time = time.time()
        print(f"DEBUG: Ação DELETE concluída em {end_time - start_time:.2f} segundos.")
        return {"statusCode": status_code, "body": json.dumps(result)}

    # ---- AÇÃO SEARCH (sem alterações) ----
    elif action == "search":
        # ... (código igual ao anterior)
        prompt = event.get("prompt", ""); keywords_param = event.get("keywords")
        match_type = event.get("match_type", "or").lower(); max_results = int(event.get("max_results", 10))
        full = event.get("full", False) in [True, 'true', 'True', 1]
        search_terms = []
        if prompt and isinstance(prompt, str) and prompt.strip(): search_terms.append(prompt.strip())
        if isinstance(keywords_param, list): search_terms.extend([k.strip() for k in keywords_param if isinstance(k, str) and k.strip()])
        elif isinstance(keywords_param, str) and keywords_param.strip(): search_terms.append(keywords_param.strip())
        if not search_terms: return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "Nenhum termo de busca (prompt/keywords) válido."})}
        print(f"DEBUG: Ação SEARCH com termos={search_terms}, match_type='{match_type}', max_results={max_results}, full={full}")
        try:
            results = search_documents(search_terms, match_type, max_results, full) # Já usa REST para query
            end_time = time.time()
            print(f"DEBUG: Ação SEARCH concluída em {end_time - start_time:.2f} segundos. {len(results)} resultados.")
            return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "results": results})}
        except Exception as e:
            print(f"ERRO durante a busca: {e}"); print(traceback.format_exc())
            return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno durante a busca: {e}"})}

    # ---- AÇÃO LISTIDS (sem alterações) ----
    elif action == "listids":
        # ... (código igual ao anterior)
        keywords = event.get("keywords", []); match_type = event.get("match_type", "or").lower()
        if isinstance(keywords, str): keywords = [keywords]
        elif not isinstance(keywords, list): keywords = []
        valid_keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        print(f"DEBUG: Ação LISTIDS com keywords={valid_keywords}")
        try:
             list_max_results = int(event.get("max_results", 100))
             results = search_documents(valid_keywords if valid_keywords else [""], match_type, max_results=list_max_results, full=False) # Já usa REST para query
             search_ids = [r["id"] for r in results if "id" in r]
             end_time = time.time()
             print(f"DEBUG: Ação LISTIDS concluída em {end_time - start_time:.2f} segundos. {len(search_ids)} IDs.")
             return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "search_ids": search_ids})}
        except Exception as e:
            print(f"ERRO durante listids: {e}"); print(traceback.format_exc())
            return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno durante listids: {e}"})}

    # ---- AÇÃO INVÁLIDA (sem alterações) ----
    else:
        print(f"ERRO: Ação inválida recebida: '{action}'")
        end_time = time.time()
        print(f"DEBUG: Requisição com ação inválida concluída em {end_time - start_time:.2f} segundos.")
        return {
            "statusCode": 400,
            "body": json.dumps({"status": "error", "message": f"Ação inválida: '{action}'. Use 'insert', 'delete', 'search' ou 'listids'."})
        }

# ----------------- Inicialização da Lambda -----------------
# Carrega o índice do S3/local quando a Lambda "acorda"
load_index()