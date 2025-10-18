# -*- coding: utf-8 -*-
# InsertDynamo.py

import boto3
import unicodedata
from datetime import datetime
import time
import traceback
import os
import json
from botocore.exceptions import ClientError
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
import re

# --- CONSTANTS & CONFIG ---
LOG_PREFIX = "[InsertDynamo]"
DEFAULT_LISTA_Publication_PK_VALUE = "#listaPublication"
# Standardize to lowercase if desired
DEFAULT_LIST_ATTRIBUTE_NAME = "titulos" # Changed from TitulosSet
DEFAULT_COMPOSITE_KEY_SEPARATOR = '#' # Using # as separator

# --- BOTO3 UTILITIES ---
serializer = TypeSerializer()
deserializer = TypeDeserializer()

# === HELPER FUNCTION TO CONVERT SETS FOR JSON ---
def convert_sets(obj):
    if isinstance(obj, set): return list(obj)
    if isinstance(obj, dict): return {k: convert_sets(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_sets(item) for item in obj]
    return obj

# === NORMALIZATION / FORMATTING HELPERS ===
def normalize_key(text):
    """Normalizes text for keys/search. NFD, lowercase, no diacritics, strip, collapse space."""
    if text is None: return None
    try:
        text = str(text)
        normalized = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower().strip()
        normalized = ' '.join(normalized.split())
        return normalized if normalized else None
    except Exception as e:
        print(f"{LOG_PREFIX} [WARN] Error during normalize_key for '{text}': {e}. Using simple lower/strip.")
        normalized = str(text).lower().strip()
        return normalized if normalized else None

def format_sortable_date(date_str):
    """Formats various date strings into YYYY-MM-DD. Returns None if parsing fails."""
    if date_str is None: return None
    original_str = str(date_str).strip()
    try: return datetime.strptime(original_str, "%Y-%m-%d").strftime('%Y-%m-%d')
    except ValueError: pass

    supported_formats = ["%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%y", "%y-%m-%d", "%d.%m.%Y", "%Y.%m.%d", "%Y"]
    for fmt in supported_formats:
        try:
            if fmt == "%Y" and len(original_str) == 4 and original_str.isdigit():
                 return datetime.strptime(original_str, fmt).strftime('%Y-01-01')
            elif fmt != "%Y":
                dt = datetime.strptime(original_str, fmt)
                year = dt.year
                if year < 100:
                   year = year + 2000 if year < 70 else year + 1900
                return f"{year:04d}-{dt.month:02d}-{dt.day:02d}"
        except (ValueError, TypeError): continue
    print(f"{LOG_PREFIX} [WARN] Could not parse date '{original_str}' into YYYY-MM-DD. Returning None.")
    return None

def split_and_normalize(text, delimiter=','):
    """Splits string, normalizes each part, returns list of non-empty strings."""
    if text is None: return []
    if isinstance(text, (list, set)):
        items_to_process = text
    elif isinstance(text, str):
        items_to_process = text.split(delimiter)
    else:
        items_to_process = [str(text)]
    items = [normalize_key(item) for item in items_to_process]
    return [item for item in items if item]

# === GENERAL DYNAMODB HELPERS ===
def clean_dynamodb_item(item_dict, partition_key_name, sort_key_name):
    """
    Cleans dict for DynamoDB: Checks PK/SK, removes Nones/empty strings/empty lists/sets,
    strips strings. Raises ValueError if PK/SK become empty.
    """
    if not isinstance(item_dict, dict):
        raise TypeError("Input must be a dictionary.")
    cleaned_item = {}
    key_attributes = {partition_key_name, sort_key_name}
    for key_name in key_attributes:
        key_value = item_dict.get(key_name)
        if key_value is None or (isinstance(key_value, str) and not key_value.strip()):
            raise ValueError(f"Critical DB Error: Primary key '{key_name}' cannot be empty or None.")
        cleaned_item[key_name] = key_value.strip() if isinstance(key_value, str) else key_value
    for k, v in item_dict.items():
        if k in key_attributes: continue
        if isinstance(v, str):
            stripped_v = v.strip()
            if stripped_v: cleaned_item[k] = stripped_v
        elif isinstance(v, (list, set)):
            cleaned_collection = []
            for item in v:
                if isinstance(item, str):
                    stripped_item = item.strip()
                    if stripped_item: cleaned_collection.append(stripped_item)
                elif item is not None: cleaned_collection.append(item)
            if cleaned_collection:
                 cleaned_item[k] = list(set(cleaned_collection)) # Store unique as list
        elif isinstance(v, dict):
             if v: cleaned_item[k] = v
        elif v is not None:
             cleaned_item[k] = v
    return cleaned_item

def convert_dict_to_dynamodb_item(py_dict):
    """Converts a Python dictionary to DynamoDB attribute-value format."""
    try:
        if not py_dict: raise ValueError("Cannot serialize an empty dictionary.")
        return {k: serializer.serialize(v) for k, v in py_dict.items()}
    except (TypeError, AttributeError) as e:
        print(f"{LOG_PREFIX} [ERROR] DynamoDB Serialization Error for dict: {py_dict}")
        raise TypeError(f"DynamoDB Serialization Error: {e}")
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Unexpected DynamoDB Serialization Error for dict: {py_dict}")
        raise

def _create_index_transact_item(operation: str, prefix: str, suffix_norm: str, composite_sort_key_ref: str, table_name: str, pk_name: str, sk_name: str):
    """Helper for index items. PK=Prefix#NormValue, SK=CompositeKey (Main Item PK)."""
    if not prefix or not suffix_norm or not composite_sort_key_ref:
        print(f"{LOG_PREFIX} [WARN] Skipping index item {operation} due to missing components: prefix='{prefix}', suffix='{suffix_norm}', sk_ref='{composite_sort_key_ref}'")
        return None, None
    index_pk = f"{prefix.upper()}#{suffix_norm}"
    index_sk = str(composite_sort_key_ref)
    if not index_pk.strip() or not index_sk.strip():
         print(f"{LOG_PREFIX} [WARN] Skipping index item {operation} due to empty generated key: PK='{index_pk}', SK='{index_sk}'")
         return None, None
    index_item_key_dict = { pk_name: index_pk, sk_name: index_sk }
    try:
        dynamodb_index_item_key = convert_dict_to_dynamodb_item(index_item_key_dict)
        if operation == 'Put':
            return {'Put': {'TableName': table_name, 'Item': dynamodb_index_item_key}}, index_item_key_dict
        elif operation == 'Delete':
             return {'Delete': {'TableName': table_name, 'Key': dynamodb_index_item_key}}, index_item_key_dict
        else:
             raise ValueError(f"Invalid operation '{operation}' specified for index item.")
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Failed creating index transact item ({operation}) for PK='{index_pk}', SK='{index_sk}': {e}")
        return None, index_item_key_dict


# === ATOMIC UPDATE FUNCTION (ADJUSTED FOR NEW PAYLOAD & NAMING) ===
def handle_dynamodb_update_atomic(
    payload: dict,
    dynamodb_table_name: str,
    dynamodb_client,
    partition_key_name: str,
    sort_key_name: str,
    list_pk_value: str, # e.g., "#listaPublication"
    list_attr_name: str # e.g., "titulos"
):
    """
    Performs an atomic UPDATE/INSERT using TransactWriteItems based on NEW payload format.
    - Stores the main document item (PK=CompositeID, SK=FormattedDate) with attributes
      named in lowercase_with_underscores.
    - Creates/Updates associated index items based on NORMALIZED values.
    - Adds the Original Title to the specified set attribute of the publication list item.

    Args:
        payload (dict): Dictionary containing the data for the item. Expected keys include:
                        'titulo', 'nome_Publication', 'data', 'tipo_publication' (optional),
                        'especialidade' (optional), 'autor' (optional),
                        'categoria1' (optional), 'categoria2' (optional), 'paginas' (optional).
                        Can also include other fields like 's3KeyPdf', 's3KeyHtml', 'lastIndexed'.
        dynamodb_table_name (str): Name of the DynamoDB table.
        dynamodb_client: Initialized low-level DynamoDB client.
        partition_key_name (str): Name of the table's partition key attribute (e.g., 'PK').
        sort_key_name (str): Name of the table's sort key attribute (e.g., 'SK').
        list_pk_value (str): Partition key value for the publication list item (e.g., '#listaPublication').
        list_attr_name (str): Attribute name holding the set of titles in the list item (e.g., 'titulos').

    Returns: dict: Summary of the update attempt.
    Raises: ValueError, ClientError, RuntimeError.
    """
    print(f"{LOG_PREFIX} Starting handle_dynamodb_update_atomic (New Payload Format) for table: {dynamodb_table_name}")

    # --- Extract Original Values from New Payload Structure ---
    # Use .get() for safe access, store originals for the main item
    titulo_original = payload.get("titulo")
    # NOTE: Payload example showed "nome_Publication" (mixed case), ensure consistency
    nome_publication_original = payload.get("nome_Publication")
    data_original = payload.get("data")
    tipo_publication_original = payload.get("tipo_publication") # Optional
    especialidade_original_str = payload.get("especialidade")     # Optional, may be comma-separated
    autor_original_str = payload.get("autor")                 # Optional, may be comma-separated
    categoria1_original_str = payload.get("categoria1")           # Optional, may be comma-separated
    categoria2_original_str = payload.get("categoria2")           # Optional, may be comma-separated
    paginas_original = payload.get("paginas")                 # Optional

    # Extract potential additional fields from payload
    s3_key_pdf = payload.get("s3KeyPdf")
    s3_key_html = payload.get("s3KeyHtml")
    last_indexed = payload.get("lastIndexed")
    # Add any other fields you expect in the payload here

    separator = DEFAULT_COMPOSITE_KEY_SEPARATOR # Use the defined separator

    # --- Validate Required Fields for Key Construction ---
    # Based on new payload keys needed for generating the main item PK and SK
    required_fields = { "titulo": titulo_original, "data": data_original, "nome_Publication": nome_publication_original }
    missing = [f for f, v in required_fields.items() if v is None or (isinstance(v, str) and not str(v).strip())]
    if missing:
        raise ValueError(f"DB Update Input Error: Required values missing from payload for key generation: {', '.join(missing)}")

    # --- Generate Keys ---
    # Main Item PK: Composite of original Title and Pub Name
    main_item_pk_composite = f"{str(titulo_original).strip()}{separator}{str(nome_publication_original).strip()}"
    # Main Item SK: Formatted Date
    main_item_sk_formatted_date = format_sortable_date(data_original)
    # List Item SK: Original Pub Name (as provided in payload)
    list_item_sk_original_pub_name = nome_publication_original # Use the raw name

    # Validate generated keys
    if not main_item_pk_composite: raise ValueError(f"DB Update Key Error: Main Item PK (Composite ID) is empty.")
    if not main_item_sk_formatted_date: raise ValueError(f"DB Update Key Error: Main Item SK (Formatted Date) is empty or invalid for input data '{data_original}'.")
    if not list_item_sk_original_pub_name: raise ValueError(f"DB Update Key Error: List Item SK (Original Publication Name) is empty.")

    # --- Generate NORMALIZED Values *ONLY* for Index Creation ---
    # Use the normalization functions on the original string values
    especialidade_list_normalized = split_and_normalize(especialidade_original_str)
    autor_list_normalized = split_and_normalize(autor_original_str)
    categoria1_list_normalized = split_and_normalize(categoria1_original_str)
    categoria2_list_normalized = split_and_normalize(categoria2_original_str)
    normalized_tipo_publication = normalize_key(tipo_publication_original)

    print(f"{LOG_PREFIX} [DEBUG] Update - Main Item Keys: PK='{main_item_pk_composite}', SK='{main_item_sk_formatted_date}'")
    print(f"{LOG_PREFIX} [DEBUG] Update - List Item Key: PK='{list_pk_value}', SK='{list_item_sk_original_pub_name}'")
    print(f"{LOG_PREFIX} [DEBUG] Update - Normalized Values for Indexes: Esp={especialidade_list_normalized}, Aut={autor_list_normalized}, Cat1={categoria1_list_normalized}, Cat2={categoria2_list_normalized}, Tipo={normalized_tipo_publication}")

    # --- Prepare Transaction Items ---
    transact_items = []
    processed_keys_log = [] # For logging/debugging

    # 1. Prepare Main Item Put (Using ORIGINAL values but NEW attribute names)
    main_item_dict = {
        partition_key_name: main_item_pk_composite,         # PK = Composite Raw ID
        sort_key_name: main_item_sk_formatted_date,         # SK = Formatted Date
        # Use lowercase_underscore names for attributes stored in DynamoDB
        'titulo': titulo_original,                          # Store Original Title
        'nome_publication': nome_publication_original,      # Store Original Pub Name
        'data_original': data_original,                     # Store Original Date String if needed
        'tipo_publication': tipo_publication_original,      # Store Original Type String (if present)
        'especialidade': especialidade_original_str,        # Store Original Especialidade String (if present)
        'autor': autor_original_str,                        # Store Original Autor String (if present)
        'categoria1': categoria1_original_str,             # Store Original Categoria_1 String (if present)
        'categoria2': categoria2_original_str,             # Store Original Categoria_2 String (if present)
        'paginas': paginas_original,                        # Store Original Paginas value (if present)
        # Add the other extracted fields with lowercase_underscore names
        's3_key_pdf': s3_key_pdf,
        's3_key_html': s3_key_html,
        'last_indexed': last_indexed
        # Add any other attributes from the payload directly here, using the desired DynamoDB name
    }
    try:
        # Clean the dict (remove Nones, empty strings/collections, strip whitespace) before converting
        # clean_dynamodb_item raises ValueError if PK/SK are invalid
        cleaned_main_item = clean_dynamodb_item(main_item_dict.copy(), partition_key_name, sort_key_name)

        dynamodb_main_item = convert_dict_to_dynamodb_item(cleaned_main_item)
        transact_items.append({'Put': {'TableName': dynamodb_table_name, 'Item': dynamodb_main_item}})

        # Log the key being processed
        main_item_key_log = {partition_key_name: main_item_pk_composite, sort_key_name: main_item_sk_formatted_date}
        processed_keys_log.append({"operation": "Put", "type": "MainItem", "key": main_item_key_log})
        print(f"{LOG_PREFIX} [DEBUG] Prepared main item Put: Key={main_item_key_log}")

    except ValueError as ve: # Catch specific key errors from cleaning
         print(f"{LOG_PREFIX} [ERROR] Failed preparing main item Put transaction due to validation error: {ve}")
         raise # Re-raise validation errors immediately
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Failed preparing main item Put transaction for PK='{main_item_pk_composite}', SK='{main_item_sk_formatted_date}': {e}")
        # Log dictionaries only if a preparation error occurred for debugging
        print(f"{LOG_PREFIX} [ERROR] Original Dict for Main Item: {main_item_dict}")
        print(f"{LOG_PREFIX} [ERROR] Cleaned Dict for Main Item: {cleaned_main_item if 'cleaned_main_item' in locals() else 'N/A'}")
        raise RuntimeError(f"Failed preparing main item transaction: {e}")

    # 2. Prepare Index Item Puts (Using NORMALIZED values)
    #    The Sort Key for index items is the Partition Key of the main item (composite ID)
    index_map = {
        "Esp": especialidade_list_normalized,
        "Aut": autor_list_normalized,
        "Cat1": categoria1_list_normalized,
        "Cat2": categoria2_list_normalized
    }
    for prefix, norm_list in index_map.items():
        # Ensure norm_list is actually a list before iterating
        if isinstance(norm_list, list):
            for norm_val in norm_list:
                # Check if norm_val is non-empty AFTER normalization
                if norm_val:
                    idx_item_tx, idx_key_log = _create_index_transact_item(
                        operation='Put',
                        prefix=prefix,
                        suffix_norm=norm_val,
                        composite_sort_key_ref=main_item_pk_composite, # Link back to main item PK
                        table_name=dynamodb_table_name,
                        pk_name=partition_key_name,
                        sk_name=sort_key_name
                    )
                    if idx_item_tx:
                        transact_items.append(idx_item_tx)
                        processed_keys_log.append({"operation": "Put", "type": f"Index_{prefix}", "key": idx_key_log})
                        print(f"{LOG_PREFIX} [DEBUG] Prepared Index Put {prefix}: Key={idx_key_log}")
                    # else: # Helper function logs errors/warnings


    # Handle Tipo index separately if it exists and is not empty
    if normalized_tipo_publication:
        idx_item_tx, idx_key_log = _create_index_transact_item(
            operation='Put',
            prefix="Tipo", # Use consistent prefix casing
            suffix_norm=normalized_tipo_publication,
            composite_sort_key_ref=main_item_pk_composite, # Link back to main item PK
            table_name=dynamodb_table_name,
            pk_name=partition_key_name,
            sk_name=sort_key_name
        )
        if idx_item_tx:
            transact_items.append(idx_item_tx)
            processed_keys_log.append({"operation": "Put", "type": "Index_Tipo", "key": idx_key_log})
            print(f"{LOG_PREFIX} [DEBUG] Prepared Index Put Tipo: Key={idx_key_log}")
        # else: # Helper function logs errors/warnings


    # 3. Prepare List Item Update (Add original title to the set attribute)
    #    PK = list_pk_value (e.g., "#listaPublication")
    #    SK = Original Publication Name (from payload)
    list_item_key_dict = {
        partition_key_name: list_pk_value,
        sort_key_name: list_item_sk_original_pub_name # Use the original publication name
    }
    try:
        # Validate key components before conversion
        if not list_pk_value or not list_item_sk_original_pub_name:
             raise ValueError(f"List item key components are missing: PK='{list_pk_value}', SK='{list_item_sk_original_pub_name}'")

        dynamodb_list_item_key = convert_dict_to_dynamodb_item(list_item_key_dict)

        # Value to add must be the original title string, within a Set structure for ADD
        title_to_add = str(titulo_original).strip()
        if not title_to_add:
             raise ValueError(f"Cannot add empty or invalid title '{titulo_original}' to the list item set.")

        # DynamoDB ADD action for String Set (SS)
        value_to_add_set = {'SS': [title_to_add]}

        # Use Update with ADD operation on the Set attribute (list_attr_name, e.g., 'titulos')
        update_item_tx = {
            'Update': {
                'TableName': dynamodb_table_name,
                'Key': dynamodb_list_item_key,
                'UpdateExpression': f"ADD #listAttr :valueToAddSet",
                'ExpressionAttributeNames': {'#listAttr': list_attr_name}, # Use the provided attribute name
                'ExpressionAttributeValues': {':valueToAddSet': value_to_add_set}
                # Consider ConditionExpression: 'attribute_exists(#pkAttr)'?
            }
        }
        transact_items.append(update_item_tx)

        # Log the key and value being processed
        processed_keys_log.append({
            "operation": "Update (Add to Set)",
            "type": "ListItem",
            "key": list_item_key_dict,
            "attributeUpdated": list_attr_name,
            "value_added": title_to_add
        })
        print(f"{LOG_PREFIX} [DEBUG] Prepared list item Update (ADD): Key={list_item_key_dict}, Attribute='{list_attr_name}', Title='{title_to_add}'")

    except ValueError as ve:
         print(f"{LOG_PREFIX} [ERROR] Failed preparing list item update transaction due to validation error: {ve}")
         raise # Re-raise specific validation errors
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Failed preparing list item update transaction Key={list_item_key_dict}, Attribute='{list_attr_name}': {e}")
        raise RuntimeError(f"Failed preparing list item update transaction: {e}")

    # --- Execute Transaction ---
    if not transact_items:
        raise ValueError(f"{LOG_PREFIX} DB Error Tx-Update: No transaction items were successfully prepared for main item PK='{main_item_pk_composite}'. Check preparation errors above.")

    if len(transact_items) > 100:
        raise ValueError(f"{LOG_PREFIX} DB Error Tx-Update: Number of items ({len(transact_items)}) exceeds DynamoDB transaction limit (100) for main item PK='{main_item_pk_composite}'. Consider redesign.")

    print(f"{LOG_PREFIX} [INFO] Attempting atomic transaction with {len(transact_items)} items for main item PK='{main_item_pk_composite}'.")
    # print(f"{LOG_PREFIX} [DEBUG] Transaction Items: {json.dumps(transact_items, indent=2)}") # Verbose Debugging

    try:
        response = dynamodb_client.transact_write_items(TransactItems=transact_items)
        status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        print(f"{LOG_PREFIX} DB Tx-Update Success for Composite ID: '{main_item_pk_composite}'. Status: {status_code}")

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        cancellation_reasons = e.response.get('CancellationReasons')

        print(f"{LOG_PREFIX} [ERROR] DB Tx-Update ClientError [{error_code}]: {error_message} for main item PK='{main_item_pk_composite}'")
        if cancellation_reasons:
            print(f"{LOG_PREFIX} [ERROR] Transaction Cancellation Reasons:")
            for i, reason in enumerate(cancellation_reasons):
                reason_code = reason.get('Code', 'N/A')
                reason_msg = reason.get('Message', 'N/A')
                item_details = "N/A"
                if i < len(transact_items):
                    try: item_details = json.dumps(transact_items[i])
                    except Exception: item_details = f"Could not serialize item {i} details."
                print(f"  - Item {i}: Code='{reason_code}', Message='{reason_msg}'")
                # print(f"    Failed Item Details: {item_details}") # Uncomment with caution
        else:
            print(f"{LOG_PREFIX} [ERROR] Full ClientError Response: {e.response}")
        raise # Re-raise the original exception

    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] DB Tx-Update Unexpected Error for main item PK='{main_item_pk_composite}': {e}")
        print(traceback.format_exc())
        raise RuntimeError(f"DB Tx Update Unexpected Server Error: {str(e)}")

    # --- Success ---
    success_msg = f"DB Atomic Update Success: Transaction ({len(transact_items)} items) completed for main item PK='{main_item_pk_composite}'."
    print(f"{LOG_PREFIX} [INFO] {success_msg}")

    return {
        "message": success_msg,
        "transactionItemsAttempted": len(transact_items),
        "processedKeys": processed_keys_log # Return the log of keys for debugging/confirmation
    }


# === ATOMIC REMOVAL FUNCTION (ADJUSTED FOR NEW PAYLOAD & NAMING) ===
def handle_dynamodb_removal_atomic(
    payload: dict,
    dynamodb_table_name: str,
    dynamodb_client,
    partition_key_name: str,
    sort_key_name: str,
    list_pk_value: str,
    list_attr_name: str, # e.g., "titulos"
    separator: str = DEFAULT_COMPOSITE_KEY_SEPARATOR
):
    """
    Performs an atomic REMOVAL using TransactWriteItems for a single content item.
    - Fetches the main item using Composite Key to get its Sort Key (Date).
    - Deletes the main document item (PK=CompositeID, SK=Date).
    - Deletes associated index items based on normalized values FROM THE NEW PAYLOAD FORMAT.
    - Deletes the Original Title from the specified set attribute of the publication list item.

    Args:
        payload (dict): Must contain the ORIGINAL values needed to reconstruct the main PK
                        and find index items (e.g., 'titulo', 'nome_Publication',
                        'tipo_publication', 'especialidade', etc., using the NEW lowercase format).
                        *** 'data' is NO LONGER required in the payload. ***
        dynamodb_table_name (str): Name of the DynamoDB table.
        dynamodb_client: Initialized low-level DynamoDB client.
        partition_key_name (str): Name of the table's partition key attribute.
        sort_key_name (str): Name of the table's sort key attribute.
        list_pk_value (str): Partition key value for the publication list item.
        list_attr_name (str): Attribute name holding the set of titles (e.g., 'titulos').
        separator (str): Separator used in the composite key.

    Returns: dict: Summary of the deletion attempt.
    Raises: ValueError, ClientError, RuntimeError.
    """
    print(f"{LOG_PREFIX} Starting ATOMIC REMOVAL (fetch SK internally, New Payload) for table: {dynamodb_table_name}")

    # --- Extract Payload Attributes (Need originals - USING NEW PAYLOAD FORMAT) ---
    titulo_original = payload.get("titulo")                 # Using new key 'titulo'
    nome_publication_original = payload.get("nome_Publication") # Using new key 'nome_Publication'
    tipo_publication_orig = payload.get("tipo_publication") # Using new key 'tipo_publication'
    especialidade_str_orig = payload.get("especialidade")     # Using new key 'especialidade'
    autor_str_orig = payload.get("autor")                 # Using new key 'autor'
    categoria1_str_orig = payload.get("categoria1")           # Using new key 'categoria1'
    categoria2_str_orig = payload.get("categoria2")           # Using new key 'categoria2'

    # --- Validate REQUIRED fields for key reconstruction (using NEW keys) ---
    required_fields = { "titulo": titulo_original, "nome_Publication": nome_publication_original }
    missing = [f for f, v in required_fields.items() if v is None or (isinstance(v, str) and not str(v).strip())]
    if missing:
        raise ValueError(f"DB Atomic Removal Input Error: Required values missing from payload: {', '.join(missing)}")

    # --- Reconstruct Keys needed (Part 1) ---
    composite_id_pk_main = f"{str(titulo_original).strip()}{separator}{str(nome_publication_original).strip()}"
    original_nome_publication_sk_list = nome_publication_original

    if not composite_id_pk_main: raise ValueError(f"DB Atomic Removal Error: Composite ID (PK Main) became empty.")
    if not original_nome_publication_sk_list: raise ValueError(f"DB Atomic Removal Error: Original Publication Name (SK List) became empty.")

    # --- Step 1: Fetch the main item to get its Sort Key (Data) ---
    fetched_sk_main = None
    print(f"{LOG_PREFIX} [DEBUG] Removal - Fetching item to get SK. Querying PK='{composite_id_pk_main}'")
    try:
        query_response = dynamodb_client.query(
            TableName=dynamodb_table_name,
            KeyConditionExpression=f"#pk = :pkval",
            ExpressionAttributeNames={'#pk': partition_key_name},
            ExpressionAttributeValues={':pkval': {'S': composite_id_pk_main}}, Limit=1
        )
        items_dynamo = query_response.get('Items', [])
        if not items_dynamo:
            msg = f"DB Atomic Removal Warning: Main item not found for PK='{composite_id_pk_main}'. Cannot get SK. Assuming already deleted or never existed."
            print(f"{LOG_PREFIX} {msg}")
            return { "message": msg, "transactionItemsAttempted": 0, "processedKeys": [], "status": "NOT_FOUND" }
        item_py = {k: deserializer.deserialize(v) for k, v in items_dynamo[0].items()}
        fetched_sk_main = item_py.get(sort_key_name)
        if fetched_sk_main is None:
             raise ValueError(f"DB Atomic Removal Error: Sort Key '{sort_key_name}' not found in fetched item for PK='{composite_id_pk_main}'. Item Data: {item_py}")
        fetched_sk_main = str(fetched_sk_main)
        print(f"{LOG_PREFIX} [DEBUG] Removal - Successfully fetched item. Found SK ('{sort_key_name}') = '{fetched_sk_main}'")
    except ClientError as e:
        print(f"{LOG_PREFIX} [ERROR] DB Atomic Removal: ClientError while fetching item SK for PK='{composite_id_pk_main}': {e}")
        raise
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] DB Atomic Removal: Unexpected error while fetching item SK for PK='{composite_id_pk_main}': {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"Unexpected error fetching item SK: {e}")

    # --- Regenerate NORMALIZED keys for index items ---
    especialidade_list_normalized = split_and_normalize(especialidade_str_orig)
    autor_list_normalized = split_and_normalize(autor_str_orig)
    categoria1_list_normalized = split_and_normalize(categoria1_str_orig)
    categoria2_list_normalized = split_and_normalize(categoria2_str_orig)
    normalized_tipo_publication = normalize_key(tipo_publication_orig)

    print(f"{LOG_PREFIX} [DEBUG] Removal - Normalized Keys for Index Deletion (from new payload): Esp={especialidade_list_normalized}, Aut={autor_list_normalized}, Cat1={categoria1_list_normalized}, Cat2={categoria2_list_normalized}, Tipo={normalized_tipo_publication}")

    # --- Prepare Transaction Items for Deletion ---
    transact_items = []; processed_keys_log = []

    # 1. Main Item Delete
    main_item_key_dict = { partition_key_name: composite_id_pk_main, sort_key_name: fetched_sk_main }
    try:
        dynamodb_main_item_key = convert_dict_to_dynamodb_item(main_item_key_dict)
        transact_items.append({'Delete': {'TableName': dynamodb_table_name, 'Key': dynamodb_main_item_key}})
        processed_keys_log.append({"operation": "Delete", "type": "MainItem", "key": main_item_key_dict})
        print(f"{LOG_PREFIX} [DEBUG] Removal - Prepared Main Item Delete: Key={main_item_key_dict}")
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Failed prep main item delete Key={main_item_key_dict}: {e}")
        raise RuntimeError(f"Failed prep main item delete Key={main_item_key_dict}: {e}")

    # 2. Index Items Delete
    index_map = { "Esp": especialidade_list_normalized, "Aut": autor_list_normalized, "Cat1": categoria1_list_normalized, "Cat2": categoria2_list_normalized }
    for prefix, norm_list in index_map.items():
        if isinstance(norm_list, list):
            for norm_val in norm_list:
                if norm_val:
                    idx_item_tx, idx_key_log = _create_index_transact_item('Delete', prefix, norm_val, composite_id_pk_main, dynamodb_table_name, partition_key_name, sort_key_name)
                    if idx_item_tx:
                        transact_items.append(idx_item_tx)
                        processed_keys_log.append({"operation": "Delete", "type": f"Index_{prefix}", "key": idx_key_log})
                        print(f"{LOG_PREFIX} [DEBUG] Removal - Prepared Index Delete {prefix}: Key={idx_key_log}")

    if normalized_tipo_publication:
        idx_item_tx, idx_key_log = _create_index_transact_item('Delete', "Tipo", normalized_tipo_publication, composite_id_pk_main, dynamodb_table_name, partition_key_name, sort_key_name)
        if idx_item_tx:
            transact_items.append(idx_item_tx)
            processed_keys_log.append({"operation": "Delete", "type": "Index_Tipo", "key": idx_key_log})
            print(f"{LOG_PREFIX} [DEBUG] Removal - Prepared Index Delete Tipo: Key={idx_key_log}")

    # 3. List/Set Update Item (DELETE Original Title)
    list_item_key_dict = { partition_key_name: list_pk_value, sort_key_name: original_nome_publication_sk_list }
    try:
        title_to_remove = str(titulo_original).strip()
        if not title_to_remove: raise ValueError("Cannot remove empty title from list item set.")
        dynamodb_list_item_key = convert_dict_to_dynamodb_item(list_item_key_dict)
        value_to_remove_set = {'SS': [title_to_remove]}
        update_item_tx = {
            'Update': {
                'TableName': dynamodb_table_name, 'Key': dynamodb_list_item_key,
                'UpdateExpression': f"DELETE #listAttr :valueToRemoveSet",
                'ExpressionAttributeNames': {'#listAttr': list_attr_name},
                'ExpressionAttributeValues': {':valueToRemoveSet': value_to_remove_set},
            }
        }
        transact_items.append(update_item_tx)
        processed_keys_log.append({
            "operation": "Update (Delete from Set)", "type": "ListItem",
            "key": list_item_key_dict, "attributeUpdated": list_attr_name,
            "value_removed": title_to_remove
        })
        print(f"{LOG_PREFIX} [DEBUG] Removal - Prepared list item Update (DELETE): Key={list_item_key_dict}, Attribute='{list_attr_name}', Title='{title_to_remove}'")
    except ValueError as ve:
         print(f"{LOG_PREFIX} [ERROR] Failed preparing list item delete update due to validation: {ve}")
         raise
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Failed prep list item delete update Key={list_item_key_dict}, Attribute='{list_attr_name}': {e}")
        raise RuntimeError(f"Failed prep list item delete update: {e}")

    # --- Perform Atomic Transaction ---
    if not transact_items:
        print(f"{LOG_PREFIX} [ERROR] DB Atomic Removal: No transaction items prepared for PK='{composite_id_pk_main}'. Main item delete likely failed preparation.")
        raise RuntimeError(f"DB Atomic Removal: No transaction items prepared for PK='{composite_id_pk_main}'.")
    if len(transact_items) > 100:
        raise ValueError(f"Cannot remove atomically: Tx size ({len(transact_items)}) exceeds limit for main item PK='{composite_id_pk_main}'.")

    print(f"{LOG_PREFIX} [INFO] Attempting removal transaction with {len(transact_items)} items for main item PK='{composite_id_pk_main}'.")
    try:
        response = dynamodb_client.transact_write_items(TransactItems=transact_items)
        status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        print(f"{LOG_PREFIX} DB Tx-Removal Success for main item PK: '{composite_id_pk_main}'. Status: {status_code}")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        cancellation_reasons = e.response.get('CancellationReasons')
        print(f"{LOG_PREFIX} [ERROR] DB Tx-Removal ClientError [{error_code}]: {error_message} for main PK='{composite_id_pk_main}'")
        if cancellation_reasons:
            print(f"{LOG_PREFIX} [ERROR] Transaction Cancellation Reasons:")
            for i, reason in enumerate(cancellation_reasons):
                reason_code = reason.get('Code', 'N/A'); reason_msg = reason.get('Message', 'N/A')
                print(f"  - Item {i}: Code='{reason_code}', Message='{reason_msg}'")
        else: print(f"{LOG_PREFIX} [ERROR] Full ClientError Response: {e.response}")
        raise
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] DB Tx-Removal Unexpected: {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"DB Tx Removal Unexpected Server Error: {str(e)}")

    success_msg = f"DB Atomic Removal Success: Transaction completed for main PK '{composite_id_pk_main}'. Targeted {len(processed_keys_log)} operations."
    print(f"{LOG_PREFIX} [INFO] {success_msg}")
    return { "message": success_msg, "transactionItemsAttempted": len(transact_items), "processedKeys": processed_keys_log, "status": "DELETED" }


# === CREATE JSON FUNCTION (MODIFIED OUTPUT STRUCTURE) ===
# (No changes needed here as it doesn't take the payload directly)
def normalize_publication_name_for_s3_py(name: str) -> str:
    """Normalizes publication name for S3 filename."""
    if not isinstance(name, str): return ''
    try:
        normalized = name.lower()
        normalized = unicodedata.normalize('NFD', normalized).encode('ascii', 'ignore').decode('utf-8')
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        normalized = normalized.strip('_')
        return normalized
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Error during normalization of '{name}': {e}")
        return ''

def create_json_from_publication(
    publication_name: str,
    dynamodb_table_name: str,
    dynamodb_client,
    s3_client,
    s3_bucket_name: str,
    partition_key_name: str,
    sort_key_name: str,
    list_pk_value: str,
    list_attr_name: str, # e.g., 'titulos'
    separator: str = DEFAULT_COMPOSITE_KEY_SEPARATOR
):
    """Fetches docs, transforms, normalizes pub name for S3 key, uploads JSON."""
    print(f"{LOG_PREFIX} Starting create_json_from_publication for: '{publication_name}'")
    start_time = time.time()
    if not publication_name or not publication_name.strip():
        raise ValueError("CreateJSON Error: 'publication_name' cannot be empty.")
    if not s3_bucket_name:
        raise EnvironmentError("CreateJSON Error: S3_BUCKET_NAME is not configured.")

    transformed_publication_data = {}
    processed_titles, failed_titles, throttled_reads = 0, 0, 0

    # 1. Get list of original titles
    list_item_key = { partition_key_name: {'S': list_pk_value}, sort_key_name: {'S': publication_name} }
    print(f"{LOG_PREFIX} [DEBUG] Getting list item with Key: {list_item_key}")
    try: response = dynamodb_client.get_item(TableName=dynamodb_table_name, Key=list_item_key)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == 'ProvisionedThroughputExceededException': throttled_reads += 1
        print(f"{LOG_PREFIX} [ERROR] ClientError getting list item: [{error_code}] {e.response['Error']['Message']}")
        raise
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Unexpected error getting list item: {e}")
        raise RuntimeError(f"Unexpected Server Error getting list item: {str(e)}")

    if 'Item' not in response:
        msg = f"Publication list item not found for PK='{list_pk_value}', SK='{publication_name}'."
        print(f"{LOG_PREFIX} [WARN] {msg}")
        return {"message": msg, "titles_processed": 0, "s3_path": None, "status": "NOT_FOUND"}
    list_item_dynamo = response['Item']
    if list_attr_name not in list_item_dynamo:
        msg = f"Attribute '{list_attr_name}' not found in list item for '{publication_name}'."
        print(f"{LOG_PREFIX} [WARN] {msg}")
        return {"message": msg, "titles_processed": 0, "s3_path": None, "status": "ATTRIBUTE_MISSING"}
    try:
        original_titles = deserializer.deserialize(list_item_dynamo[list_attr_name])
        if not isinstance(original_titles, set) or not original_titles:
             msg = f"Attribute '{list_attr_name}' is not a non-empty set for '{publication_name}'."
             print(f"{LOG_PREFIX} [WARN] {msg}")
             return {"message": msg, "titles_processed": 0, "s3_path": None, "status": "EMPTY_OR_INVALID_SET"}
        print(f"{LOG_PREFIX} [INFO] Found {len(original_titles)} original titles for '{publication_name}'.")
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Failed to deserialize '{list_attr_name}': {e}")
        raise RuntimeError(f"Failed to process title list: {str(e)}")

    # 2. Iterate through titles, fetch main item, transform
    for original_title in original_titles:
        original_title_str = str(original_title).strip()
        if not original_title_str:
            print(f"{LOG_PREFIX} [WARN] Skipping empty title found in the set."); failed_titles += 1; continue
        composite_id_pk = f"{original_title_str}{separator}{publication_name}"
        print(f"{LOG_PREFIX} [DEBUG] Querying for main item with PK: '{composite_id_pk}'")
        try:
            query_response = dynamodb_client.query(
                TableName=dynamodb_table_name,
                KeyConditionExpression=f"#pk = :pkval",
                ExpressionAttributeNames={'#pk': partition_key_name},
                ExpressionAttributeValues={':pkval': {'S': composite_id_pk}}, Limit=1
            )
            items_dynamo = query_response.get('Items', [])
            if not items_dynamo:
                print(f"{LOG_PREFIX} [WARN] Main item not found via query for PK='{composite_id_pk}'. Skipping."); failed_titles += 1; continue
            item_py = {k: deserializer.deserialize(v) for k, v in items_dynamo[0].items()}
            transformed_item = item_py.copy()
            if sort_key_name in transformed_item: transformed_item['Data'] = transformed_item.pop(sort_key_name)
            if partition_key_name in transformed_item: transformed_item.pop(partition_key_name)
            transformed_publication_data[original_title_str] = transformed_item
            processed_titles += 1
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == 'ProvisionedThroughputExceededException': throttled_reads += 1
            else: print(f"{LOG_PREFIX} [ERROR] ClientError querying PK='{composite_id_pk}': [{error_code}] {e.response['Error']['Message']}")
            failed_titles += 1; continue
        except Exception as e:
            print(f"{LOG_PREFIX} [ERROR] Unexpected error querying/processing PK='{composite_id_pk}': {e}")
            failed_titles += 1; continue

    print(f"{LOG_PREFIX} [INFO] Fetched & transformed data. Processed: {processed_titles}, Failed/Skipped: {failed_titles}, Throttled Reads: {throttled_reads}")
    if not transformed_publication_data:
        msg = f"No documents successfully retrieved/transformed for '{publication_name}'."
        print(f"{LOG_PREFIX} [WARN] {msg}")
        return {"message": msg, "titles_processed": 0, "s3_path": None, "status": "NO_DATA_RETRIEVED"}

    # 3. Prepare JSON, NORMALIZE name for S3 key, and upload
    try:
        final_data_for_json = convert_sets(transformed_publication_data)
        json_output = json.dumps(final_data_for_json, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Failed to serialize final data to JSON for '{publication_name}': {e}")
        raise RuntimeError(f"Failed to create JSON output: {str(e)}")
    normalized_s3_filename_base = normalize_publication_name_for_s3_py(publication_name)
    if not normalized_s3_filename_base:
         error_msg = f"Normalization of publication name '{publication_name}' resulted in unusable filename. Cannot upload to S3."
         print(f"{LOG_PREFIX} [ERROR] {error_msg}"); raise ValueError(error_msg)
    s3_key = f"JSON/{normalized_s3_filename_base}.json"
    print(f"{LOG_PREFIX} [INFO] Uploading transformed JSON to s3://{s3_bucket_name}/{s3_key} (Normalized from '{publication_name}')")
    try:
        s3_client.put_object(Bucket=s3_bucket_name, Key=s3_key, Body=json_output.encode('utf-8'), ContentType='application/json; charset=utf-8')
        print(f"{LOG_PREFIX} [INFO] Successfully uploaded transformed JSON to S3.")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown'); error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"{LOG_PREFIX} [ERROR] S3 Upload ClientError: {error_code} - {error_message}")
        raise
    except Exception as e:
        print(f"{LOG_PREFIX} [ERROR] Unexpected error during S3 upload: {str(e)}")
        raise RuntimeError(f"Unexpected S3 Upload Error: {str(e)}")

    end_time = time.time(); duration = round((end_time - start_time) * 1000)
    success_msg = f"Successfully created and uploaded JSON for '{publication_name}' to s3://{s3_bucket_name}/{s3_key}"
    print(f"{LOG_PREFIX} {success_msg} (Duration: {duration}ms)")
    return {
        "message": success_msg, "s3_bucket": s3_bucket_name, "s3_key": s3_key,
        "titles_processed": processed_titles, "titles_failed_or_skipped": failed_titles,
        "throttled_reads_logged": throttled_reads, "processing_time_ms": duration,
        "status": "SUCCESS"
    }