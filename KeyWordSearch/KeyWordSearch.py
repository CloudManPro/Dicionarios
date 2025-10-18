# -*- coding: utf-8 -*-
import os
import time
import json
import zipfile
import boto3
import re
import shutil
from whoosh.index import exists_in, create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.analysis import StandardAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup, AndGroup
from whoosh import qparser as whoosh_qparser
from whoosh.query import Term, And, Or, Not, Every
from collections import defaultdict
# Suppress the specific SyntaxWarning from whoosh codec if desired
import warnings
warnings.filterwarnings("ignore", message="\"is\" with 'int' literal. Did you mean \"==\"?", category=SyntaxWarning, module="whoosh.codec.whoosh3")


# Configurações do S3 e caminhos locais
S3_BUCKET = os.getenv("AWS_S3_BUCKET_TARGET_NAME_0", "your-default-bucket-name")
ZIP_KEY = "IndexParts_v2.zip"
TMP_INDEX_DIR = "/tmp/index_parts_v2"

# Variável global para armazenar o índice carregado
ix = None
s3 = boto3.client('s3')

# --- Mapeamento de Nomes de Campo ---
# Mapeia os nomes recebidos na API (chaves) para os nomes usados no schema Whoosh (valores)
API_TO_SCHEMA_MAP = {
    "nome_Publication": "publication_name",
    "especialidade": "specialty",
    "autor": "author",
    "categoria1": "category1",
    "categoria2": "category2"
    # Adicione outros mapeamentos se necessário
}

# Mapeia os nomes do schema Whoosh (chaves) de volta para os nomes da API (valores)
SCHEMA_TO_API_MAP = {v: k for k, v in API_TO_SCHEMA_MAP.items()}


# --- Schema Definition (Usa nomes internos do Whoosh) ---
def get_schema():
    """Retorna o schema Whoosh SEM publication_date e pages."""
    return Schema(
        part_doc_id=ID(stored=True, unique=True),
        original_doc_id=ID(stored=True),
        part_name=ID(stored=True),
        content=TEXT(analyzer=StandardAnalyzer(), stored=True),
        # Nomes internos do Whoosh
        publication_name=ID(stored=True),
        specialty=KEYWORD(stored=True, commas=True, lowercase=True, scorable=True),
        author=KEYWORD(stored=True, commas=True, lowercase=True, scorable=True),
        category1=KEYWORD(stored=True, commas=True, lowercase=True, scorable=True),
        category2=KEYWORD(stored=True, commas=True, lowercase=True, scorable=True)
    )

# --- Funções Auxiliares (Download/Upload - Sem mudanças aqui) ---
def _download_and_extract_index():
    """Baixa e extrai o índice do S3."""
    if not os.path.exists(TMP_INDEX_DIR):
        try: os.makedirs(TMP_INDEX_DIR); print(f"Diretório {TMP_INDEX_DIR} criado.")
        except OSError as e: print(f"Erro ao criar diretório {TMP_INDEX_DIR}: {e}"); return False
    zip_path = os.path.join("/tmp", os.path.basename(ZIP_KEY))
    if os.path.exists(zip_path):
        try: os.remove(zip_path)
        except OSError as e: print(f"Erro ao remover zip antigo {zip_path}: {e}")
    if os.path.exists(TMP_INDEX_DIR):
        try: shutil.rmtree(TMP_INDEX_DIR); os.makedirs(TMP_INDEX_DIR)
        except Exception as e: print(f"Aviso: Falha ao limpar {TMP_INDEX_DIR}: {e}")
    try:
        print(f"Baixando s3://{S3_BUCKET}/{ZIP_KEY} para {zip_path}")
        s3.download_file(S3_BUCKET, ZIP_KEY, zip_path); print(f"Download concluído.")
        print(f"Extraindo {zip_path} para {TMP_INDEX_DIR}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(TMP_INDEX_DIR)
        print("Índice extraído com sucesso."); os.remove(zip_path)
        return True
    except Exception as e:
        error_code = None
        if hasattr(e, 'response') and 'Error' in e.response: error_code = e.response['Error'].get('Code')
        if error_code == 'NoSuchBucket': print(f"Erro: Bucket S3 '{S3_BUCKET}' não encontrado."); return False
        elif error_code == '404' or error_code == 'NoSuchKey': print(f"Zip s3://{S3_BUCKET}/{ZIP_KEY} não encontrado. Novo índice será criado."); return False
        elif error_code == 'AccessDenied': print(f"Erro: Acesso Negado ao baixar s3://{S3_BUCKET}/{ZIP_KEY}."); return False
        else: print(f"Erro S3 (Code: {error_code}) ao baixar/extrair: {e}"); return False
        print(f"Erro geral ao baixar/extrair: {e}"); return False

def load_index_globally():
    """Carrega ou cria o índice com o SCHEMA atualizado."""
    global ix; schema = get_schema(); download_successful = _download_and_extract_index()
    if download_successful and exists_in(TMP_INDEX_DIR):
        try:
            print(f"Abrindo índice baixado {TMP_INDEX_DIR}..."); ix = open_dir(TMP_INDEX_DIR, schema=schema)
            if ix.schema != schema:
                 print("ALERTA: Schema diferente! Recriando."); ix.close(); shutil.rmtree(TMP_INDEX_DIR); os.makedirs(TMP_INDEX_DIR); ix = create_in(TMP_INDEX_DIR, schema)
                 print("Novo índice recriado (incompatibilidade schema).")
            else: print("Índice baixado aberto (schema compatível).")
        except Exception as e:
            print(f"Erro ao abrir índice baixado: {e}. Criando novo."); ix = None
            try:
                 if os.path.exists(TMP_INDEX_DIR): shutil.rmtree(TMP_INDEX_DIR)
                 os.makedirs(TMP_INDEX_DIR); ix = create_in(TMP_INDEX_DIR, schema); print("Novo índice criado pós-erro.")
            except Exception as ce: print(f"Erro CRÍTICO ao recriar índice: {ce}"); ix = None
    elif not exists_in(TMP_INDEX_DIR):
         print(f"Índice não encontrado. Criando novo..."); ix = None
         try:
            if not os.path.exists(TMP_INDEX_DIR): os.makedirs(TMP_INDEX_DIR)
            elif os.listdir(TMP_INDEX_DIR): shutil.rmtree(TMP_INDEX_DIR); os.makedirs(TMP_INDEX_DIR)
            ix = create_in(TMP_INDEX_DIR, schema); print("Novo índice criado.")
         except Exception as e: print(f"Erro CRÍTICO ao criar índice inicial: {e}"); ix = None
    elif not download_successful and exists_in(TMP_INDEX_DIR):
        print(f"Download falhou, tentando abrir índice local...");
        try:
            ix = open_dir(TMP_INDEX_DIR, schema=schema)
            if ix.schema != schema:
                 print("ALERTA: Schema local diferente. Recriando."); ix.close(); shutil.rmtree(TMP_INDEX_DIR); os.makedirs(TMP_INDEX_DIR); ix = create_in(TMP_INDEX_DIR, schema)
                 print("Novo índice recriado (incompatibilidade local).")
            else: print("Índice local aberto (schema compatível).")
        except Exception as e:
             print(f"Erro ao abrir local: {e}. Criando novo."); ix = None
             try: shutil.rmtree(TMP_INDEX_DIR); os.makedirs(TMP_INDEX_DIR); ix = create_in(TMP_INDEX_DIR, schema); print("Novo índice criado pós-erro local.")
             except Exception as ce: print(f"Erro CRÍTICO ao recriar índice: {ce}"); ix = None
    else: print("Erro: Estado inesperado."); ix = None

def save_index():
    """Compacta e envia o índice para o S3."""
    global ix;
    if not ix: print("Erro: Tentando salvar índice, mas ix é None."); return
    zip_path = os.path.join("/tmp", os.path.basename(ZIP_KEY)); start_time = time.time(); index_was_open_and_closed_by_us = False
    try:
        if ix:
            print("Fechando índice para salvamento...");
            try: ix.close(); index_was_open_and_closed_by_us = True; print("Índice fechado.")
            except Exception as close_err: print(f"Erro ao fechar índice: {close_err}")
        else: print("Aviso crítico: save_index, ix é None."); return
        if not os.path.isdir(TMP_INDEX_DIR): print(f"Erro: Diretório {TMP_INDEX_DIR} não encontrado."); return

        print(f"Compactando índice de {TMP_INDEX_DIR}...");
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(TMP_INDEX_DIR):
                files = [f for f in files if not f.endswith(('.lock', '.lk'))]
                for file in files:
                    file_path = os.path.join(root, file); rel_path = os.path.relpath(file_path, TMP_INDEX_DIR)
                    if os.path.exists(file_path): zipf.write(file_path, rel_path)
        compaction_time = time.time() - start_time
        if os.path.exists(zip_path):
             zip_size_mb = os.path.getsize(zip_path)/1024/1024; print(f"Compactação concluída {compaction_time:.2f}s ({zip_size_mb:.2f} MB)")
             if zip_size_mb > 0:
                 print(f"Enviando {zip_path} para s3://{S3_BUCKET}/{ZIP_KEY}..."); s3.upload_file(zip_path, S3_BUCKET, ZIP_KEY)
                 upload_time = time.time() - start_time - compaction_time; print(f"Enviado ao S3 em {upload_time:.2f}s.")
             else: print("Aviso: Zip vazio.")
             try: os.remove(zip_path)
             except OSError as e: print(f"Aviso: Falha ao remover {zip_path}: {e}")
        else: print("Erro: Zip não foi criado.")
    except Exception as e: print(f"Erro CRÍTICO durante save_index: {e}")
    finally:
        if index_was_open_and_closed_by_us:
             try:
                print("Recarregando índice no finally..."); load_index_globally()
                if ix: print("Índice recarregado.")
                else: print("Erro: Falha ao recarregar índice no finally.")
             except Exception as final_e: print(f"Erro CRÍTICO fatal no finally: {final_e}"); ix = None
        if os.path.exists(zip_path):
            try: os.remove(zip_path)
            except OSError as e: print(f"Aviso: Falha remover {zip_path} final: {e}")

# --- Carregamento Inicial ---
print(f"Iniciando carregamento global do índice v2 de {TMP_INDEX_DIR}...")
load_index_globally()
if ix:
    try: print(f"Carregamento inicial concluído. Índice contém {ix.doc_count()} partes.")
    except Exception as e: print(f"Erro count inicial: {e}.")
else: print("Carregamento inicial FALHOU.");

# --- Handler da Lambda ---
def lambda_handler(event, context):
    global ix; headers = {
        "Content-Type": "application/json", "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
    }
    if event.get('httpMethod') == 'OPTIONS': return {'statusCode': 200, 'headers': headers, 'body': json.dumps('Success')}
    print("Evento:", json.dumps(event))
    body_data = {};
    if isinstance(event.get('body'), str):
        try: body_data = json.loads(event['body'])
        except json.JSONDecodeError: return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "Corpo JSON inválido."}), "headers": headers}
    elif isinstance(event, dict) and 'httpMethod' not in event: body_data = event
    elif isinstance(event.get('body'), dict): body_data = event['body']
    is_warmup = (event.get("source") == "aws.events" or event.get("action") == "warmup" or body_data.get("action") == "warmup")
    if is_warmup:
        print("Warmup.");
        if ix:
            try: cnt = ix.doc_count(); print(f"Índice OK ({cnt})."); return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "message": f"OK ({cnt})."}), "headers": headers}
            except Exception as e: print(f"Erro índice warmup: {e}."); return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro índice warmup: {e}"}), "headers": headers}
        else:
            print("Warmup: ix é None. Recarregando..."); load_index_globally()
            if ix: cnt = ix.doc_count(); print("Recarregado OK."); return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "message": f"Recarregado OK ({cnt})."}), "headers": headers}
            else: print("Erro: Recarga falhou."); return {"statusCode": 500, "body": json.dumps({"status": "error", "message": "Índice não carregado."}), "headers": headers}
    if ix is None:
        print("Erro Crítico: ix é None. Recarregando..."); load_index_globally()
        if ix is None: print("Recarga falhou. 503."); return {"statusCode": 503, "body": json.dumps({"status": "error", "message": "Índice não disponível."}), "headers": headers}
        else: print(f"Índice recarregado (Doc count: {ix.doc_count()}).")
    action = body_data.get("action", "").lower();
    if not action: action = event.get('queryStringParameters', {}).get('action', '').lower()
    print("Ação:", action)

    # --- AÇÃO INSERT ---
    if action == "insert":
        original_doc_id = body_data.get("original_doc_id")
        metadata = body_data.get("metadata") # Dict com nomes da API: {"autor": ..., "especialidade": ...}
        parts = body_data.get("parts")
        if not original_doc_id or not isinstance(metadata, dict) or not isinstance(parts, list) or not parts:
            return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "Payload inválido insert."}), "headers": headers}
        try:
            writer = ix.writer()
            deleted_count = writer.delete_by_query(Term("original_doc_id", str(original_doc_id)))
            print(f"{deleted_count} partes antigas removidas para {original_doc_id}.")
            indexed_parts_count = 0
            for part_data in parts:
                part_name = part_data.get("part_name"); part_content = part_data.get("text")
                if not part_name or part_content is None: continue
                part_doc_id = f"{str(original_doc_id)}_{str(part_name)}"
                doc_to_index = {
                    "part_doc_id": part_doc_id, "original_doc_id": str(original_doc_id),
                    "part_name": str(part_name), "content": str(part_content),
                    # --- MAPEAMENTO API -> SCHEMA ---
                    # Chave é nome do SCHEMA, valor vem do METADATA usando nome da API
                    "publication_name": str(metadata.get(SCHEMA_TO_API_MAP.get("publication_name", "nome_Publication"), "")), # Usa get no map p/ segurança
                    "specialty": str(metadata.get(SCHEMA_TO_API_MAP.get("specialty", "especialidade"), "")),
                    "author": str(metadata.get(SCHEMA_TO_API_MAP.get("author", "autor"), "")),
                    "category1": str(metadata.get(SCHEMA_TO_API_MAP.get("category1", "categoria1"), "")),
                    "category2": str(metadata.get(SCHEMA_TO_API_MAP.get("category2", "categoria2"), "")),
                }
                writer.add_document(**doc_to_index); indexed_parts_count += 1
            writer.commit(); print(f"Doc {original_doc_id} indexado ({indexed_parts_count}). Salvando..."); save_index()
            return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "message": f"Doc {original_doc_id} indexado ({indexed_parts_count})."}), "headers": headers}
        except Exception as e: print(f"Erro indexar {original_doc_id}: {e}"); import traceback; traceback.print_exc(); return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno indexar: {e}"}), "headers": headers}

    # --- AÇÃO DELETE ---
    elif action == "delete":
        original_doc_id = body_data.get("original_doc_id")
        if not original_doc_id: return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "'original_doc_id' obrigatório."}), "headers": headers}
        try:
            writer = ix.writer(); deleted_count = writer.delete_by_query(Term("original_doc_id", str(original_doc_id))); writer.commit()
            if deleted_count > 0: print(f"Doc {original_doc_id} removido ({deleted_count}). Salvando..."); save_index(); return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "message": f"Doc {original_doc_id} ({deleted_count}) removido."}), "headers": headers}
            else: return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "message": f"Doc {original_doc_id} não encontrado."}), "headers": headers}
        except Exception as e: print(f"Erro remover {original_doc_id}: {e}"); return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno remover: {e}"}), "headers": headers}

    # --- AÇÃO SEARCH ---
    elif action == "search":
        query_str = body_data.get("query")
        filters = body_data.get("filters", {}) # Espera filtros com NOMES DA API: {"autor": ..., "especialidade": ...}
        limit_per_doc = 1000; final_limit = 10
        if not query_str: return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "'query' obrigatório."}), "headers": headers}
        if not isinstance(filters, dict): return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "'filters' deve ser objeto."}), "headers": headers}
        try: limit_str = body_data.get("limit", "10"); final_limit = int(limit_str); assert 0 < final_limit <= 100
        except: final_limit = 10
        try:
            start_search_time = time.time()
            with ix.searcher() as searcher:
                schema = searcher.schema
                content_parser = QueryParser("content", schema=schema, group=AndGroup); main_query = content_parser.parse(query_str)
                filter_queries = []
                # --- Processa filtros de PARTES (se houver) ---
                include_parts = filters.get("include_parts"); exclude_parts = filters.get("exclude_parts")
                if isinstance(include_parts, list) and include_parts: filter_queries.append(Or([Term("part_name", str(p)) for p in include_parts]))
                elif isinstance(exclude_parts, list) and exclude_parts: filter_queries.append(Not(Or([Term("part_name", str(p)) for p in exclude_parts])))

                # --- Processa filtros de METADADOS (usando mapeamento) ---
                print(f"Filtros recebidos (API): {filters}")
                for api_key, filter_value in filters.items():
                    # Pula chaves que não são filtros de metadados conhecidos (ex: include_parts)
                    if api_key not in API_TO_SCHEMA_MAP: continue
                    if not filter_value: continue # Ignora filtros vazios

                    # Obtém o nome do campo no schema Whoosh correspondente à chave da API
                    schema_field_name = API_TO_SCHEMA_MAP.get(api_key)
                    if not schema_field_name: # Segurança: não deveria acontecer se api_key está no map
                        print(f"Aviso: Chave API '{api_key}' sem mapeamento de schema correspondente.")
                        continue

                    print(f"Processando filtro API '{api_key}' -> Schema '{schema_field_name}' com valor '{filter_value}'")
                    is_keyword = schema_field_name in ["specialty", "author", "category1", "category2"]

                    # Constrói a query Whoosh usando o nome do SCHEMA
                    if is_keyword and isinstance(filter_value, str) and ',' in filter_value:
                         terms = [Term(schema_field_name, term.strip().lower()) for term in filter_value.split(',') if term.strip()]
                         if terms: filter_queries.append(And(terms)); print(f"  -> Query AND para {schema_field_name}: {terms}")
                    else:
                         term_text = str(filter_value).lower() if is_keyword else str(filter_value)
                         filter_queries.append(Term(schema_field_name, term_text)); print(f"  -> Query Term para {schema_field_name}: '{term_text}'")
                # --- Fim processamento de filtros ---

                final_query = And([main_query] + filter_queries) if filter_queries else main_query
                print(f"Query Whoosh final: {final_query}")

                aggregated_results = defaultdict(lambda: {"aggregated_score": 0.0, "metadata": {}, "matched_parts": []})
                results = searcher.search(final_query, limit=limit_per_doc, scored=True)

                for hit in results:
                    original_id = hit["original_doc_id"]; aggregated_results[original_id]["aggregated_score"] += hit.score
                    if not aggregated_results[original_id]["metadata"]:
                         stored_fields = hit.fields()
                         # --- MAPEAMENTO SCHEMA -> API para METADADOS DE SAÍDA ---
                         api_metadata = {"original_doc_id": original_id} # Inclui sempre
                         for schema_key, api_key in SCHEMA_TO_API_MAP.items():
                              if schema_key in stored_fields:
                                   api_metadata[api_key] = stored_fields[schema_key]
                         aggregated_results[original_id]["metadata"] = api_metadata
                    # Armazena dados da parte (sem mapeamento necessário, usa nomes internos)
                    part_info = hit.fields(); part_info["score"] = hit.score
                    aggregated_results[original_id]["matched_parts"].append(part_info)

            sorted_aggregated = sorted(aggregated_results.values(), key=lambda x: x["aggregated_score"], reverse=True)
            final_results = sorted_aggregated[:final_limit]
            search_time = time.time() - start_search_time
            print(f"Busca concluída {search_time:.2f}s. Retornando {len(final_results)} docs.")
            # Retorna metadados com NOMES DA API
            return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "results": final_results}, ensure_ascii=False), "headers": headers}
        except whoosh_qparser.QueryParserError as qpe: print(f"Erro sintaxe query '{query_str}': {qpe}"); return {"statusCode": 400, "body": json.dumps({"status": "error", "message": f"Erro sintaxe query: {qpe}"}), "headers": headers}
        except Exception as e: print(f"Erro CRÍTICO busca: {e}"); import traceback; traceback.print_exc(); return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno busca: {e}"}), "headers": headers}

    # --- AÇÃO LISTIDS ---
    elif action == "listids":
        print("Executando ação 'listids'...")
        try:
            with ix.searcher() as searcher:
                results_with_filters = [] # Conterá objetos com nomes da API
                unique_original_ids = set()
                reader = searcher.reader();
                for doc_id_bytes in reader.lexicon("original_doc_id"): unique_original_ids.add(doc_id_bytes.decode('utf-8'))
                print(f"Listids: {len(unique_original_ids)} IDs únicos. Buscando metadados...")
                for original_id in sorted(list(unique_original_ids)):
                    results = searcher.search(Term("original_doc_id", original_id), limit=1)
                    if results:
                        hit = results[0]; stored_fields = hit.fields()
                        # --- MAPEAMENTO SCHEMA -> API para FILTROS DE SAÍDA ---
                        # Constrói o dict de filtros usando os nomes da API como chaves
                        api_filters_data = {}
                        for schema_key, api_key in SCHEMA_TO_API_MAP.items():
                             if schema_key in stored_fields: # Pega valor usando chave do schema
                                  api_filters_data[api_key] = stored_fields[schema_key] # Guarda usando chave da API
                        results_with_filters.append({
                            "original_doc_id": original_id,
                            # "filters" agora contém chaves da API
                            "filters": api_filters_data
                        })
            print(f"Listids concluída. Retornando {len(results_with_filters)} IDs.")
            # Retorna 'filters' com NOMES DA API
            return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "ids_with_filters": results_with_filters}, ensure_ascii=False), "headers": headers}
        except Exception as e: print(f"Erro CRÍTICO listids: {e}"); import traceback; traceback.print_exc(); return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno listar IDs: {e}"}), "headers": headers}

    # --- AÇÃO LISTPARTIDS ---
    elif action == "listpartids":
        print("Executando ação 'listpartids'...")
        try:
            with ix.searcher() as searcher:
                part_ids = set(); reader = searcher.reader()
                for part_id_bytes in reader.lexicon("part_doc_id"): part_ids.add(part_id_bytes.decode('utf-8'))
            print(f"Listpartids: {len(part_ids)} IDs de parte únicos.")
            return {"statusCode": 200, "body": json.dumps({"status": "sucesso", "part_ids": sorted(list(part_ids))}, ensure_ascii=False), "headers": headers}
        except Exception as e: print(f"Erro CRÍTICO listpartids: {e}"); return {"statusCode": 500, "body": json.dumps({"status": "error", "message": f"Erro interno listar IDs parte: {e}"}), "headers": headers}

    # --- AÇÃO INVÁLIDA ---
    else:
        print(f"Erro: Ação inválida: '{action}'")
        valid = ["insert", "delete", "search", "listids", "listpartids", "warmup"]
        return {"statusCode": 400, "body": json.dumps({"status": "error", "message": f"Ação inválida: '{action}'. Use: {', '.join(valid)}."}), "headers": headers}