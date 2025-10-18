# -*- coding: utf-8 -*-
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
import traceback

# ----------------- Configurações -----------------
S3_BUCKET = os.getenv("AWS_S3_BUCKET_TARGET_NAME_0", "your-default-bucket-name")
INDEX_ZIP_KEY = "Semantic2K.zip"
TMP_INDEX_DIR = "/tmp/index"
INDEX_FILE = os.path.join(TMP_INDEX_DIR, "index.pkl")

# --- Configuração da API do Google GenAI ---
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCdOsmdzquKHolKf6WdWhFNkbRMAfkdF_E") 

# --- Modelos ---
MODEL_NAME_EMBEDDING = "text-embedding-004"
MODEL_NAME_COUNTING = "gemini-1.5-flash-latest"

# --- Nomes API REST ---
MODEL_NAME_EMBEDDING_API = f"models/{MODEL_NAME_EMBEDDING}"
MODEL_NAME_COUNTING_API = f"models/{MODEL_NAME_COUNTING}"

# --- Endpoints REST ---
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_URL_EMBED_CONTENT = f"{API_BASE_URL}/{MODEL_NAME_EMBEDDING_API}:embedContent"
API_URL_BATCH_EMBED = f"{API_BASE_URL}/{MODEL_NAME_EMBEDDING_API}:batchEmbedContents"
API_URL_COUNT_TOKENS = f"{API_BASE_URL}/{MODEL_NAME_COUNTING_API}:countTokens"

MAX_TOKENS_PER_CHUNK = 2040
TOKENS_SEPARADOR_ESTIMADO = 2

# --- Logging Prefix ---
LOG_PREFIX = "[RAGAppV2.14-BE-FixedSearchKeys]" # Fixed Search Key Handling

print(f"{LOG_PREFIX} DEBUG: Lambda starting...")
print(f"{LOG_PREFIX} DEBUG: S3_BUCKET = {S3_BUCKET}")
print(f"{LOG_PREFIX} DEBUG: INDEX_ZIP_KEY = {INDEX_ZIP_KEY}")
print(f"{LOG_PREFIX} DEBUG: MODEL_NAME_EMBEDDING = {MODEL_NAME_EMBEDDING}")
print(f"{LOG_PREFIX} DEBUG: MODEL_NAME_COUNTING = {MODEL_NAME_COUNTING}")
print(f"{LOG_PREFIX} DEBUG: API Key Loaded: {'Yes' if API_KEY and API_KEY != 'YOUR_GOOGLE_API_KEY' else 'NO / USING DEFAULT PLACEHOLDER!!'}")

s3 = boto3.client('s3')
index = {} # Global index dictionary

# ----------------- Funções Auxiliares -----------------
def strip_html_tags(text):
    if not isinstance(text, str): return ""
    return re.sub('<[^<]+?>', '', text)

def build_request_url(base_url):
    if not API_KEY or API_KEY == 'YOUR_GOOGLE_API_KEY': raise ValueError("API Key do Google não configurada.")
    return f"{base_url}?key={API_KEY}"

# ----- count_tokens_func_rest -----
def count_tokens_func_rest(text_to_count):
    if not API_KEY or API_KEY == 'YOUR_GOOGLE_API_KEY': print(f"{LOG_PREFIX} ERRO: API Key Google não configurada."); return float('inf')
    if not text_to_count or not isinstance(text_to_count, str) or not text_to_count.strip(): return 0
    payload = {"contents": [{"parts": [{"text": text_to_count}]}]}
    headers = {"Content-Type": "application/json"}; data = json.dumps(payload).encode("utf-8")
    url = build_request_url(API_URL_COUNT_TOKENS); req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200: error_body = response.read().decode('utf-8', errors='ignore'); print(f"{LOG_PREFIX} ERRO API {response.status} (countTokens): {error_body}"); return float('inf')
            response_data = response.read().decode("utf-8"); result = json.loads(response_data)
            return int(result.get("totalTokens", float('inf'))) # Safer get
    except urllib.error.HTTPError as e: error_body = e.read().decode('utf-8', errors='ignore'); print(f"{LOG_PREFIX} DEBUG: HTTP {e.code} count_tokens: {e.reason}\nBody: {error_body}"); return float('inf')
    except Exception as e: print(f"{LOG_PREFIX} ERRO: count_tokens: {str(e)}"); print(traceback.format_exc()); return float('inf')

# ----- split_text_into_chunks_by_tokens -----
def split_text_into_chunks_by_tokens(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    print(f"{LOG_PREFIX} DEBUG: Iniciando split_text max_tokens={max_tokens}")
    final_chunks = []
    if not text or not text.strip(): print(f"{LOG_PREFIX} DEBUG: Texto vazio."); return []
    # Split by double+ newlines, keeping delimiters
    parts = re.split(r'(\n{2,})', text); paragraphs = []
    current_paragraph = ""
    for i, part in enumerate(parts): # Reconstruct paragraphs including original newlines
        if i % 2 == 0: current_paragraph += part # Text part
        else: current_paragraph += part; paragraphs.append(current_paragraph.strip()); current_paragraph = "" # Delimiter part
    if current_paragraph.strip(): paragraphs.append(current_paragraph.strip())
    if not paragraphs: paragraphs = [text.strip()] # Fallback if no double newlines
    print(f"{LOG_PREFIX} DEBUG: {len(paragraphs)} parágrafos iniciais.")
    paragraphs_with_tokens = []
    print(f"{LOG_PREFIX} DEBUG: Contando tokens parag. (modelo: {MODEL_NAME_COUNTING})...")
    start_count_time = time.time()
    for i, p in enumerate(paragraphs):
        token_count = count_tokens_func_rest(p)
        if token_count == float('inf'): print(f"{LOG_PREFIX} ERRO: Falha contar tokens parag {i}. Ignorado."); continue
        paragraphs_with_tokens.append({'text': p, 'tokens': token_count})
        if token_count > max_tokens: print(f"{LOG_PREFIX} !!! ALERTA: Parag {i} ({token_count} tokens) > max {max_tokens}!")
    end_count_time = time.time(); print(f"{LOG_PREFIX} DEBUG: Contagem tokens ({len(paragraphs_with_tokens)} parag.) em {end_count_time - start_count_time:.2f} seg.")
    current_chunk_paragraphs = []; current_chunk_tokens_estimated = 0
    for item in paragraphs_with_tokens:
        paragraph_text = item['text']; paragraph_tokens = item['tokens']
        if paragraph_tokens > max_tokens: # Handle paragraphs already over limit
            if current_chunk_paragraphs: final_chunks.append("\n\n".join(current_chunk_paragraphs)) # Finalize previous chunk
            final_chunks.append(paragraph_text); print(f"{LOG_PREFIX} DEBUG: Parag. grande ({paragraph_tokens} tokens) adicionado separado.")
            current_chunk_paragraphs = []; current_chunk_tokens_estimated = 0; continue
        tokens_separador = 0 if not current_chunk_paragraphs else TOKENS_SEPARADOR_ESTIMADO
        potential_tokens = current_chunk_tokens_estimated + tokens_separador + paragraph_tokens
        if potential_tokens <= max_tokens: current_chunk_paragraphs.append(paragraph_text); current_chunk_tokens_estimated = potential_tokens
        else: # Start new chunk
            if current_chunk_paragraphs: final_chunks.append("\n\n".join(current_chunk_paragraphs))
            current_chunk_paragraphs = [paragraph_text]; current_chunk_tokens_estimated = paragraph_tokens
    if current_chunk_paragraphs: final_chunks.append("\n\n".join(current_chunk_paragraphs)) # Add last chunk
    verified_chunks = []
    print(f"{LOG_PREFIX} DEBUG: Verificando tokens {len(final_chunks)} chunks finais..."); start_verify_time = time.time()
    for i, chunk in enumerate(final_chunks):
        final_token_count = count_tokens_func_rest(chunk)
        if final_token_count == 0 or final_token_count == float('inf'): print(f"{LOG_PREFIX} AVISO: Chunk {i+1} contagem 0 ou inf. Ignorado.")
        else:
             if final_token_count > MAX_TOKENS_PER_CHUNK: print(f"{LOG_PREFIX} !!! ALERTA FINAL: Chunk {i+1} ({final_token_count} tokens) > limite {MAX_TOKENS_PER_CHUNK}!")
             verified_chunks.append(chunk) # Add even if over limit, but log warning
    end_verify_time = time.time(); print(f"{LOG_PREFIX} DEBUG: Verificação chunks em {end_verify_time - start_verify_time:.2f} seg.");
    print(f"{LOG_PREFIX} DEBUG: Retornando {len(verified_chunks)} chunks válidos."); return verified_chunks

# --- Funções de Embedding (REST API) ---
def get_embedding(text):
    if not text or not text.strip(): raise ValueError("Texto vazio para get_embedding.")
    payload = {"model": MODEL_NAME_EMBEDDING_API, "content": {"parts": [{"text": text}]}}
    headers = {"Content-Type": "application/json"}; data = json.dumps(payload).encode("utf-8")
    url = build_request_url(API_URL_EMBED_CONTENT); req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200: error_body = response.read().decode('utf-8', errors='ignore'); raise Exception(f"API Error {response.status} (embed): {error_body}")
            response_data = response.read().decode("utf-8"); result = json.loads(response_data)
            embedding_data = result.get("embedding", {})
            if "values" not in embedding_data: raise Exception(f"Resp API inesperada (emb): {result}")
            return np.array(embedding_data["values"], dtype=np.float32).flatten()
    except urllib.error.HTTPError as e: error_body = e.read().decode('utf-8', errors='ignore'); print(f"{LOG_PREFIX} DEBUG: HTTP {e.code} get_embedding: {e.reason}\nBody: {error_body}"); raise Exception(f"Erro HTTP {e.code} API emb.") from e
    except Exception as e: print(f"{LOG_PREFIX} DEBUG: Erro get_embedding: {str(e)}"); print(traceback.format_exc()); raise

def get_embeddings(texts):
    valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
    if not valid_texts: print(f"{LOG_PREFIX} DEBUG: Nenhum texto válido get_embeddings."); return []
    payload = { "requests": [ {"model": MODEL_NAME_EMBEDDING_API, "content": {"parts": [{"text": t}]}} for t in valid_texts ] }
    headers = {"Content-Type": "application/json"}; data = json.dumps(payload).encode("utf-8")
    url = build_request_url(API_URL_BATCH_EMBED); req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200: error_body = response.read().decode('utf-8', errors='ignore'); raise Exception(f"API Error {response.status} (batch): {error_body}")
            response_data = response.read().decode("utf-8"); result = json.loads(response_data)
            embeddings_list = result.get("embeddings", [])
            if len(embeddings_list) != len(valid_texts): raise Exception(f"Discrep requests/embeddings. Req={len(valid_texts)}, Emb={len(embeddings_list)}.")
            embeddings = []
            for i, embedding_data in enumerate(embeddings_list):
                if "values" not in embedding_data: raise Exception(f"Emb inválido (sem 'values') req {i}.")
                embeddings.append(np.array(embedding_data["values"], dtype=np.float32).flatten())
            print(f"{LOG_PREFIX} DEBUG: Embeddings batch (REST) count={len(embeddings)}")
            return embeddings
    except urllib.error.HTTPError as e: error_body = e.read().decode('utf-8', errors='ignore'); print(f"{LOG_PREFIX} DEBUG: HTTP {e.code} get_embeddings: {e.reason}\nBody: {error_body}"); raise Exception(f"Erro HTTP {e.code} API batch emb.") from e
    except Exception as e: print(f"{LOG_PREFIX} DEBUG: Erro get_embeddings: {str(e)}"); print(traceback.format_exc()); raise

# ----------------- Gerenciamento do índice -----------------
def load_index():
    global index
    print(f"{LOG_PREFIX} DEBUG: Iniciando load_index()")
    if not os.path.exists(TMP_INDEX_DIR):
        try: os.makedirs(TMP_INDEX_DIR); print(f"{LOG_PREFIX} DEBUG: Diretório {TMP_INDEX_DIR} criado.")
        except OSError as e: print(f"{LOG_PREFIX} ERRO: Falha ao criar {TMP_INDEX_DIR}: {e}"); return
    zip_path = "/tmp/index_download.zip"
    try:
        print(f"{LOG_PREFIX} DEBUG: Baixando {INDEX_ZIP_KEY} de {S3_BUCKET}...")
        s3.download_file(S3_BUCKET, INDEX_ZIP_KEY, zip_path)
        print(f"{LOG_PREFIX} DEBUG: Extraindo {zip_path} para {TMP_INDEX_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(TMP_INDEX_DIR)
        print(f"{LOG_PREFIX} DEBUG: Índice extraído.")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404' or 'NoSuchKey' in str(e): print(f"{LOG_PREFIX} DEBUG: Zip índice não encontrado S3.")
        else: print(f"{LOG_PREFIX} ERRO S3 ao baixar: {e}"); print(traceback.format_exc())
    except Exception as e: print(f"{LOG_PREFIX} ERRO inesperado download/extract: {e}"); print(traceback.format_exc())

    if os.path.exists(INDEX_FILE):
        print(f"{LOG_PREFIX} DEBUG: Carregando {INDEX_FILE}...")
        try:
            with open(INDEX_FILE, "rb") as f: index_loaded = pickle.load(f)
            # Processamento para garantir embeddings são numpy arrays
            processed_index = {}
            for doc_id, doc in index_loaded.items():
                 # Process doc embedding
                 if "embedding" in doc and isinstance(doc["embedding"], list):
                     try: doc["embedding"] = np.array(doc["embedding"], dtype=np.float32).flatten()
                     except ValueError as ve: print(f"{LOG_PREFIX} ERRO: Falha converter emb doc '{doc_id}'. Skip. {ve}"); doc["embedding"] = None
                 elif not isinstance(doc.get("embedding"), np.ndarray): doc["embedding"] = None # Ensure None if not ndarray
                 # Process part embeddings
                 if "parts" in doc and isinstance(doc["parts"], list):
                     for i, part in enumerate(doc["parts"]):
                         if "embedding" in part and isinstance(part["embedding"], list):
                              try: part["embedding"] = np.array(part["embedding"], dtype=np.float32)
                              except ValueError as ve_part: print(f"{LOG_PREFIX} ERRO: Falha converter part {i} emb '{doc_id}'. Skip. {ve_part}"); part["embedding"] = None
                         elif not isinstance(part.get("embedding"), np.ndarray): part["embedding"] = None
                 processed_index[doc_id] = doc
            index = processed_index # Assign processed index globally
            print(f"{LOG_PREFIX} DEBUG: Índice carregado e processado. Docs: {len(index)}")
        except (EOFError, pickle.UnpicklingError, ValueError, TypeError) as e: print(f"{LOG_PREFIX} ERRO: Falha deserializar {INDEX_FILE}: {e}. Resetando."); index = {}
        except Exception as e: print(f"{LOG_PREFIX} ERRO: Falha carregar {INDEX_FILE}: {e}"); print(traceback.format_exc()); index = {}
    else: index = {}; print(f"{LOG_PREFIX} DEBUG: {INDEX_FILE} não existe. Índice vazio.")

def save_index():
    global index
    print(f"{LOG_PREFIX} DEBUG: save_index() - Docs: {len(index)}")
    if not os.path.exists(TMP_INDEX_DIR): os.makedirs(TMP_INDEX_DIR)
    # Convert numpy arrays back to lists for pickling
    index_to_save = {}
    for doc_id, doc_data in index.items():
        saved_doc = {}
        for key, value in doc_data.items():
             if key == "embedding" and isinstance(value, np.ndarray): saved_doc[key] = value.tolist() # Convert main embedding
             elif key == "parts" and isinstance(value, list):
                 saved_parts = []
                 for part in value:
                     saved_part = part.copy() # Avoid modifying original part in memory
                     if isinstance(saved_part.get("embedding"), np.ndarray): saved_part["embedding"] = saved_part["embedding"].tolist() # Convert part embedding
                     saved_parts.append(saved_part)
                 saved_doc[key] = saved_parts
             else: saved_doc[key] = value # Copy other fields
        index_to_save[doc_id] = saved_doc
    # Save and upload
    try:
        with open(INDEX_FILE, "wb") as f: pickle.dump(index_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{LOG_PREFIX} DEBUG: Índice salvo em {INDEX_FILE}")
    except Exception as e: print(f"{LOG_PREFIX} ERRO: Falha salvar {INDEX_FILE}: {e}"); print(traceback.format_exc()); return
    zip_path = "/tmp/index_upload.zip";
    try:
        print(f"{LOG_PREFIX} DEBUG: Criando zip {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(INDEX_FILE): zipf.write(INDEX_FILE, os.path.basename(INDEX_FILE))
            else: print(f"{LOG_PREFIX} ERRO: {INDEX_FILE} não encontrado p/ zip."); return
        print(f"{LOG_PREFIX} DEBUG: Enviando zip para S3 {S3_BUCKET}/{INDEX_ZIP_KEY}")
        s3.upload_file(zip_path, S3_BUCKET, INDEX_ZIP_KEY)
        print(f"{LOG_PREFIX} DEBUG: Índice enviado S3.")
    except FileNotFoundError: print(f"{LOG_PREFIX} ERRO: Zip {zip_path} não encontrado.")
    except Exception as e: print(f"{LOG_PREFIX} ERRO: Falha zip/upload S3: {e}"); print(traceback.format_exc())

def delete_document(doc_id):
    global index
    print(f"{LOG_PREFIX} DEBUG: delete_document() - doc_id={doc_id}")
    if doc_id in index:
        del index[doc_id]; print(f"{LOG_PREFIX} DEBUG: Doc {doc_id} removido memória.")
        try: save_index(); return {"status": "sucesso", "message": f"Doc {doc_id} removido."}
        except Exception as e: print(f"{LOG_PREFIX} ERRO: Falha salvar índice após deletar {doc_id}: {e}"); raise IOError(f"Falha salvar índice após deletar {doc_id}") from e
    else: print(f"{LOG_PREFIX} DEBUG: Doc {doc_id} não encontrado."); return {"status": "error", "message": f"Doc {doc_id} não encontrado."}

# ----------------- Função de Similaridade -----------------
def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None or not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray): return 0.0
    try: vec1 = vec1.flatten(); vec2 = vec2.flatten() # Ensure 1D
    except AttributeError: return 0.0 # Should not happen with ndarray check
    if vec1.shape[0] == 0 or vec2.shape[0] == 0: return 0.0
    if vec1.shape != vec2.shape: print(f"{LOG_PREFIX} AVISO: Formatos incompatíveis cos_sim: {vec1.shape} vs {vec2.shape}"); return 0.0
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0: return 0.0
    dot_product = np.dot(vec1, vec2); similarity = dot_product / (norm1 * norm2)
    return float(np.clip(similarity, -1.0, 1.0)) # Clip to handle potential float inaccuracies

# ----------------- Função de Busca (Espera filtros com keys de armazenamento BE) -----------------
def search_documents(keywords, filters={}, match_type="or", max_results=10, full=False):
    """ Realiza busca semântica, aplicando filtros (ESPERA keys iguais às do índice). """
    # Chaves em 'filters' devem corresponder às chaves em index[doc_id]['metadata']
    # Ex: filters={'nome_Publication': '...', 'especialidade': '...'}
    print(f"{LOG_PREFIX} DEBUG: search_documents() - keywords={keywords}, filters={filters} (usa keys BE), match='{match_type}', max={max_results}, full={full}")

    if not API_KEY or API_KEY == 'YOUR_GOOGLE_API_KEY': print(f"{LOG_PREFIX} ERRO: API Key não configurada."); return []
    if not keywords or not isinstance(keywords, list) or not any(k and k.strip() for k in keywords if isinstance(k, str)): print(f"{LOG_PREFIX} DEBUG: keywords inválida ou vazia."); return []

    query_text = keywords[0].strip()
    if not query_text: print(f"{LOG_PREFIX} ERRO: Texto consulta vazio."); return []

    print(f"{LOG_PREFIX} DEBUG: Gerando embedding para consulta: '{query_text}'")
    try: query_embedding = get_embedding(query_text)
    except Exception as e: print(f"{LOG_PREFIX} ERRO: Falha gerar emb consulta '{query_text}': {e}"); return []
    if query_embedding is None: print(f"{LOG_PREFIX} ERRO: Embedding consulta None."); return [] # Should be caught by exception now

    results_list = []; all_snippets = []; docs_checked = 0; passed_filter_count = 0
    print(f"{LOG_PREFIX} DEBUG: Iniciando varredura índice ({len(index)} docs) e filtros...")
    for doc_id, doc in index.items():
        docs_checked += 1
        doc_metadata = doc.get("metadata", {}) # Metadata como armazenado no índice (keys BE)

        # --- Aplicação dos Filtros ---
        passes_filter = True
        if filters: # Filtros recebidos (com chaves BE: 'nome_Publication', 'especialidade', etc.)
            for filter_key_be, filter_value in filters.items():
                # filter_key_be AGORA é 'nome_Publication', 'especialidade', etc.
                if filter_value is not None: # Permitir filtros vazios para checar ausência? Não, só checa valor
                    doc_value = doc_metadata.get(filter_key_be) # Busca pela chave exata BE
                    # Comparação case-insensitive para strings, exata para outros
                    match = False
                    if isinstance(filter_value, str) and isinstance(doc_value, str):
                        match = filter_value.lower() == doc_value.lower()
                    elif filter_value == doc_value: # Comparação exata para não-strings
                         match = True

                    if not match:
                    #if doc_value != filter_value: # Comparação direta original (case-sensitive)
                        passes_filter = False
                        # print(f"{LOG_PREFIX} ---- FALHOU Filtro '{filter_key_be}' para Doc ID: {doc_id} (ValDoc='{doc_value}', ValFiltro='{filter_value}')") # Verbose
                        break # Sai do loop de filtros para este doc
        # --- Fim Aplicação Filtros ---

        if not passes_filter: continue # Pula para o próximo doc se falhou no filtro

        passed_filter_count += 1
        # print(f"{LOG_PREFIX} -- Doc ID: {doc_id} PASSOU nos filtros.") # Verbose

        # --- Cálculo de Similaridade (somente para docs que passaram no filtro) ---
        doc_content = doc.get("content", "")
        if full:
             doc_embedding = doc.get("embedding")
             if doc_embedding is not None:
                 doc_score = cosine_similarity(query_embedding, doc_embedding)
                 results_list.append({"id": doc_id, "content": doc_content, "score": doc_score, "metadata": doc_metadata}) # Retorna metadata com keys BE
             # else: print(f"{LOG_PREFIX} AVISO: Doc {doc_id} (passou) sem embedding (modo full).")
        else: # Busca por partes (snippets)
            doc_parts = doc.get("parts")
            part_found = False
            if isinstance(doc_parts, list) and doc_parts:
                for i, part in enumerate(doc_parts):
                    part_emb = part.get("embedding"); part_text = part.get("text", "")
                    if part_emb is not None and isinstance(part_text, str) and part_text.strip():
                        part_score = cosine_similarity(query_embedding, part_emb)
                        all_snippets.append({"doc_id": doc_id, "text": part_text, "score": part_score, "metadata": doc_metadata}) # Retorna metadata com keys BE
                        part_found = True
            # Fallback se não houver partes ou nenhuma parte válida com embedding
            if not part_found and doc.get("embedding") is not None and doc_content:
                 # print(f"{LOG_PREFIX} DEBUG: Doc {doc_id} (passou) sem partes válidas, usando emb. doc.")
                 doc_score = cosine_similarity(query_embedding, doc.get("embedding"))
                 all_snippets.append({"doc_id": doc_id, "text": doc_content, "score": doc_score, "metadata": doc_metadata}) # Retorna metadata com keys BE
    # --- Fim Loop Docs ---

    print(f"{LOG_PREFIX} DEBUG: Filtros aplicados. Checados: {docs_checked}, Passaram Filtro: {passed_filter_count}")

    # --- Processamento Final dos Resultados ---
    if full:
        results_list.sort(key=lambda x: x["score"], reverse=True); final_results = results_list[:max_results]
        print(f"{LOG_PREFIX} DEBUG: Busca 'full' retornando {len(final_results)} docs.")
        return final_results # Contém metadata com keys BE
    else:
        all_snippets.sort(key=lambda x: x["score"], reverse=True); top_n_snippets = 100; top_snippets = all_snippets[:top_n_snippets]
        grouped_results = {} # Agrupa por doc_id
        for snippet in top_snippets:
            doc_id = snippet["doc_id"]
            if doc_id not in grouped_results:
                # Inicializa com score baixo e metadata (com keys BE)
                grouped_results[doc_id] = {"id": doc_id, "score": -1.0, "content": [], "metadata": snippet["metadata"]}
            # Adiciona o texto e score da parte
            grouped_results[doc_id]["content"].append({"text": snippet["text"], "score": snippet["score"]})
            # Atualiza o score do doc para o MAIOR score de suas partes relevantes
            grouped_results[doc_id]["score"] = max(grouped_results[doc_id]["score"], snippet["score"])
        # Ordena os docs agregados pelo score máximo e limita
        final_results_list = sorted(list(grouped_results.values()), key=lambda x: x["score"], reverse=True); final_results = final_results_list[:max_results]
        print(f"{LOG_PREFIX} DEBUG: Busca snippets retornando {len(final_results)} docs agregados.")
        return final_results # Contém metadata com keys BE

# ----------------- Função Lambda Handler (NON-PROXY + Chaves BE nos Filtros) ---
def lambda_handler(event, context):
    """ Ponto de entrada da Lambda - NON-PROXY. """
    start_time = time.time()
    print(f"{LOG_PREFIX} DEBUG: Evento recebido: {json.dumps(event, indent=2)}")

    # --- API Key Check ---
    if not API_KEY or API_KEY == 'YOUR_GOOGLE_API_KEY':
        print(f"{LOG_PREFIX} ERRO FATAL: GOOGLE_API_KEY não configurada.")
        return {"status": "error", "message": "Erro interno servidor: Config API ausente."}

    # --- Warmup Check ---
    if event.get("source") == "aws.events" or event.get("WARMUP") == "TRUE":
        print(f"{LOG_PREFIX} DEBUG: Evento aquecimento.")
        try: test_count = count_tokens_func_rest("warmup"); print(f"{LOG_PREFIX} DEBUG: Teste contagem OK ({test_count} tokens).")
        except Exception as e: print(f"{LOG_PREFIX} AVISO: Exceção teste contagem: {e}")
        return {"status": "sucesso", "message": "Lambda aquecida."}

    # --- Main Logic ---
    response_body = {}
    action = event.get("action", "").lower()
    print(f"{LOG_PREFIX} DEBUG: Ação: '{action}'")

    try:
        # ---- AÇÃO INSERT (Armazena com chaves específicas BE) ----
        if action == "insert":
            doc_id = event.get("doc_id", f"doc_{int(time.time())}")
            print(f"{LOG_PREFIX} DEBUG: Ação INSERT doc_id='{doc_id}'.")

            # Extração de Texto (Prioriza 'parts[0]' se for string, senão 'content', senão 'text')
            parts_list = event.get("parts"); text_content = ""
            if isinstance(parts_list, list) and parts_list and isinstance(parts_list[0], str): text_content = parts_list[0]
            if not text_content: text_content = event.get("content", event.get("text", ""))
            if not text_content or not isinstance(text_content, str) or not text_content.strip(): raise ValueError("Conteúdo textual ('parts[0]', 'content' ou 'text') vazio ou inválido.")

            cleaned_text = strip_html_tags(text_content)
            if not cleaned_text.strip(): raise ValueError("Conteúdo vazio pós-limpeza HTML.")

            chunks = split_text_into_chunks_by_tokens(cleaned_text, MAX_TOKENS_PER_CHUNK)
            if not chunks: print(f"{LOG_PREFIX} AVISO: Nenhum chunk válido doc '{doc_id}'."); raise ValueError("Nenhum chunk válido após divisão.")

            print(f"{LOG_PREFIX} DEBUG: Gerando embeddings p/ {len(chunks)} chunks (Doc ID: {doc_id})...")
            chunk_embeddings = get_embeddings(chunks)
            if len(chunk_embeddings) != len(chunks): print(f"{LOG_PREFIX} ERRO: Discrepância batch emb doc '{doc_id}'."); raise Exception("Falha obter emb chunks.")

            combined_embedding = np.mean(chunk_embeddings, axis=0).flatten() if chunk_embeddings else None
            parts_info = [{"text": t, "embedding": e} for t, e in zip(chunks, chunk_embeddings)]

            # --- Coleta Metadata (usando chaves BE diretamente) ---
            event_metadata = event.get("metadata", {})
            if not isinstance(event_metadata, dict): print(f"{LOG_PREFIX} AVISO: 'metadata' inválido."); event_metadata = {}
            # Armazena usando as chaves EXATAS que vieram no event['metadata']
            doc_metadata = { k: v for k, v in event_metadata.items() if k in [
                "nome_Publication", "data", "especialidade", "autor", "categoria1", "categoria2"
                # Adicione outras chaves EXATAS permitidas aqui
            ] and v is not None and v != ""} # Filtra chaves permitidas e valores não vazios
            print(f"{LOG_PREFIX} DEBUG: Metadata ARMAZENADA: {doc_metadata}")
            # --- Fim Coleta Metadata ---

            # Armazena no índice usando as chaves BE ('nome_Publication', etc)
            index[doc_id] = {"id": doc_id, "content": cleaned_text, "embedding": combined_embedding, "parts": parts_info, "metadata": doc_metadata}
            print(f"{LOG_PREFIX} DEBUG: Doc '{doc_id}' adicionado/atualizado.")

            save_index()
            response_body = {"status": "sucesso", "message": f"Doc {doc_id} indexado ({len(chunks)} chunks)."}

        # ---- AÇÃO DELETE ----
        elif action == "delete":
            doc_id = event.get("id", event.get("doc_id", ""))
            print(f"{LOG_PREFIX} DEBUG: Ação DELETE doc_id='{doc_id}'")
            if not doc_id: raise ValueError("ID não fornecido para delete.")
            response_body = delete_document(doc_id) # Chama função auxiliar

        # ---- AÇÃO SEARCH (Usa chaves BE diretamente nos filtros) ----
        elif action == "search":
            prompt = event.get("prompt", ""); search_terms = [prompt.strip()] if prompt and isinstance(prompt, str) and prompt.strip() else []
            if not search_terms: raise ValueError("Termo busca ('prompt') vazio.")

            filters_from_event = event.get("filters", {}) # Filtros como chegam do FE (espera keys BE)
            if not isinstance(filters_from_event, dict):
                print(f"{LOG_PREFIX} AVISO: 'filters' inválido: {filters_from_event}. Ignorando.");
                filters_from_event = {}

            # *** AJUSTE APLICADO AQUI ***
            # Remove o mapeamento desnecessário. Usa os filtros do evento diretamente,
            # assumindo que eles já contêm as chaves de armazenamento BE ('nome_Publication', etc.)
            # Filtra apenas valores não vazios/None
            direct_filters = {k: v for k, v in filters_from_event.items() if v is not None and v != ""}
            # *** FIM DO AJUSTE ***

            max_results = int(event.get("max_results", 10))
            full = event.get("full", False) in [True, 'true', 'True', 1]

            # Log mostra os filtros DIRETOS que serão enviados para search_documents
            print(f"{LOG_PREFIX} DEBUG: Ação SEARCH prompt='{search_terms[0]}', filtros EVENTO (BE Keys)={direct_filters}, max={max_results}, full={full}")

            # *** Passa os filtros DIRETOS para search_documents ***
            results = search_documents(
                keywords=search_terms,
                filters=direct_filters, # Passa dict com chaves BE ('nome_Publication', etc.)
                max_results=max_results,
                full=full
            )
            # 'results' já contém 'metadata' com as chaves BE, conforme retornado por search_documents
            response_body = {"status": "sucesso", "results": results}

        # ---- AÇÃO LISTIDS ----
        elif action == "listids":
            print(f"{LOG_PREFIX} DEBUG: Ação LISTALLMETADATA - Iniciando scan do índice...")

            # 1. Obter filtros opcionais do evento (mesma lógica de search)
            filters_from_event = event.get("filters", {})
            if not isinstance(filters_from_event, dict):
                 print(f"{LOG_PREFIX} AVISO: 'filters' inválido em listallmetadata. Ignorando."); filters_from_event = {}
            # Usar filtros diretamente, removendo valores vazios/None
            direct_filters = {k: v for k, v in filters_from_event.items() if v is not None and v != ""}
            print(f"{LOG_PREFIX} DEBUG: Aplicando filtros (BE Keys)={direct_filters} durante scan.")

            results = []
            docs_scanned = 0
            docs_matched = 0

            if not index:
                print(f"{LOG_PREFIX} DEBUG: Índice global está vazio.")
            else:
                # 2. Iterar diretamente pelo índice global
                for doc_id, doc_data in index.items():
                    docs_scanned += 1
                    doc_metadata = doc_data.get("metadata", {}) # Pega o dict de metadata

                    # 3. Aplicar filtros (se houver) diretamente na metadata
                    passes_filter = True
                    if direct_filters:
                        for filter_key_be, filter_value in direct_filters.items():
                            doc_value = doc_metadata.get(filter_key_be)
                            match = False
                            # Comparação case-insensitive para strings, exata para outros
                            if isinstance(filter_value, str) and isinstance(doc_value, str):
                                match = filter_value.lower() == doc_value.lower()
                            elif filter_value == doc_value: # Comparação exata para não-strings ou listas/etc.
                                match = True
                            # Se o valor no doc for uma LISTA, checa se o valor do filtro está NELA (case-insensitive para strings na lista)
                            elif isinstance(doc_value, list) and isinstance(filter_value, str):
                                for item in doc_value:
                                    if isinstance(item, str) and item.lower() == filter_value.lower():
                                        match = True
                                        break
                            # Adicione outras lógicas de comparação se necessário (ex: checar se valor do filtro está contido em string do doc)

                            if not match:
                                passes_filter = False
                                break # Falhou neste filtro, vai para o próximo doc
                    # --- Fim da Aplicação de Filtros ---

                    if passes_filter:
                        docs_matched += 1
                        # 4. Adiciona o ID e sua metadata se passou nos filtros
                        results.append({
                            "id": doc_id,
                            "metadata": doc_metadata # Retorna o dict de metadata associado
                        })

            print(f"{LOG_PREFIX} DEBUG: Scan concluído. Docs Scaneados: {docs_scanned}, Docs Correspondentes (pós-filtro): {docs_matched}")
            # 5. Monta a resposta
            response_body = {
                "status": "sucesso",
                "documents": results, # Lista de dicionários {"id": ..., "metadata": ...}
                "count": len(results)
            }

        # ---- AÇÃO INVÁLIDA ----
        else:
            print(f"{LOG_PREFIX} ERRO: Ação inválida: '{action}'")
            raise ValueError(f"Ação inválida: '{action}'. Ações válidas: insert, delete, search, listids.")

    # --- Tratamento de Erros para Non-Proxy ---
    except ValueError as ve: print(f"{LOG_PREFIX} ERRO (Input): {ve}"); response_body = {"status": "error", "message": str(ve)}
    except EnvironmentError as env_err: print(f"{LOG_PREFIX} ERRO (Config): {env_err}"); response_body = {"status": "error", "message": "Erro config servidor."}
    except IOError as io_err: print(f"{LOG_PREFIX} ERRO (IO): {io_err}"); response_body = {"status": "error", "message": f"Erro interno persistir dados."}
    except Exception as e: print(f"{LOG_PREFIX} ERRO Inesperado: {e}"); print(traceback.format_exc()); response_body = {"status": "error", "message": f"Erro interno servidor."}

    end_time = time.time()
    print(f"{LOG_PREFIX} DEBUG: Requisição action='{action}' concluída em {end_time - start_time:.2f} seg.")

    # --- RETORNO NON-PROXY ---
    print(f"{LOG_PREFIX} DEBUG: Retornando objeto API GW (Non-Proxy): {json.dumps(response_body)}")
    return response_body # Retorna o dict diretamente

# ----------------- Inicialização Lambda -----------------
try:
    load_index() # Carrega o índice global na inicialização
except Exception as init_error:
    print(f"{LOG_PREFIX} ERRO CRÍTICO inicialização (load_index): {init_error}")
    print(traceback.format_exc()); index = {} # Garante que o índice está vazio se falhar