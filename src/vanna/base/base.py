r"""
Enhanced VannaBase with intelligent one-time schema training and caching.
This implementation solves the problem of requiring table names in prompts.
"""

import json
import os
import re
import sqlite3
import time
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlparse

from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import validate_config_path


class VannaBase(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)
        
        # NEW: Caching for performance
        self._schema_cache = None
        self._schema_cache_time = None
        self._cache_ttl = 3600  # Cache for 1 hour
        self._trained = False  # Track if training has been done

    def log(self, message: str, title: str = "Info"):
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        if self.language is None:
            return ""
        return f"Respond in the {self.language} language."

    # ==================== NEW: TRAINING STATUS & CACHING ====================
    
    def check_training_status(self) -> dict:
        """
        Check if the system has been trained and provide recommendations.
        Returns status information without triggering training.
        
        Returns:
            dict: Status information including training needs and statistics
        """
        if not self.run_sql_is_set:
            return {
                "status": "not_connected",
                "message": "❌ Not connected to a database",
                "needs_training": False,
                "training_count": 0,
                "table_count": 0
            }
        
        try:
            training_data = self.get_training_data()
            table_list = self._get_cached_table_list()
            
            ddl_count = len(training_data[training_data['training_data_type'] == 'ddl']) if 'training_data_type' in training_data.columns else 0
            doc_count = len(training_data[training_data['training_data_type'] == 'documentation']) if 'training_data_type' in training_data.columns else 0
            sql_count = len(training_data[training_data['training_data_type'] == 'sql']) if 'training_data_type' in training_data.columns else 0
            
            total_training = len(training_data)
            total_tables = len(table_list)
            
            if total_training == 0:
                status = "not_trained"
                message = f"⚠️  No training data found. Database has {total_tables} tables but none are trained."
                needs_training = True
            elif ddl_count < total_tables * 0.8:  # At least 80% of tables should be trained
                status = "partially_trained"
                message = f"⚠️  Partial training: {ddl_count}/{total_tables} tables trained"
                needs_training = True
            else:
                status = "trained"
                message = f"✅ Fully trained: {total_training} items ({ddl_count} tables, {doc_count} docs, {sql_count} Q&A pairs)"
                needs_training = False
            
            return {
                "status": status,
                "message": message,
                "needs_training": needs_training,
                "training_count": total_training,
                "table_count": total_tables,
                "ddl_count": ddl_count,
                "doc_count": doc_count,
                "sql_count": sql_count,
                "tables": table_list
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ Error checking training status: {e}",
                "needs_training": True,
                "training_count": 0,
                "table_count": 0
            }

    def _get_cached_table_list(self) -> List[str]:
        """
        Lightweight method to get table names with caching.
        This is fast and doesn't trigger full training.
        
        Returns:
            List[str]: List of table names in the database
        """
        # Check cache validity
        if (self._schema_cache is not None and 
            self._schema_cache_time is not None and 
            time.time() - self._schema_cache_time < self._cache_ttl):
            return self._schema_cache
        
        # Fetch table names (fast query)
        try:
            if self.dialect == "SQLite":
                tables_df = self.run_sql(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                table_list = tables_df['name'].tolist()
            elif self.dialect == "PostgreSQL":
                tables_df = self.run_sql("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                table_list = tables_df['table_name'].tolist()
            elif self.dialect == "MySQL":
                tables_df = self.run_sql("""
                    SELECT table_name 
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                """)
                table_list = tables_df['table_name'].tolist()
            elif self.dialect in ["BigQuery SQL", "Snowflake SQL"]:
                tables_df = self.run_sql("""
                    SELECT table_name 
                    FROM information_schema.tables
                """)
                table_list = tables_df['table_name'].tolist()
            elif self.dialect == "DuckDB SQL":
                tables_df = self.run_sql("""
                    SELECT table_name 
                    FROM information_schema.tables
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                """)
                table_list = tables_df['table_name'].tolist()
            else:
                return []
            
            # Update cache
            self._schema_cache = table_list
            self._schema_cache_time = time.time()
            
            return table_list
        except Exception as e:
            self.log(f"Could not fetch table list: {e}", "Warning")
            return []

    def ensure_trained(self, force: bool = False):
        """
        Ensure the system is trained. Only trains if necessary.
        
        Args:
            force (bool): Force retraining even if already trained
        """
        if self._trained and not force:
            self.log("System already trained. Use force=True to retrain.", "Info")
            return
        
        status = self.check_training_status()
        
        if force or status['needs_training']:
            self.log(f"Training required: {status['message']}", "Info")
            self.auto_train_on_schema(sample_data=True)
            self._trained = True
            self.log("Training completed!", "Success")
        else:
            self.log(status['message'], "Info")
            self._trained = True

    def show_database_info(self):
        """Display database schema information and training status"""
        if not self.run_sql_is_set:
            print("❌ Not connected to a database")
            return
        
        print("=" * 70)
        print("📊 DATABASE INFORMATION")
        print("=" * 70)
        
        try:
            # Get table list
            table_list = self._get_cached_table_list()
            print(f"\n📋 Tables ({len(table_list)}):")
            
            if self.dialect == "SQLite":
                for table in table_list[:10]:  # Show first 10
                    try:
                        cols = self.run_sql(f"PRAGMA table_info({table})")
                        col_names = cols['name'].tolist()
                        print(f"  • {table}")
                        print(f"    Columns ({len(col_names)}): {', '.join(col_names[:5])}{'...' if len(col_names) > 5 else ''}")
                    except:
                        print(f"  • {table}")
                if len(table_list) > 10:
                    print(f"  ... and {len(table_list) - 10} more tables")
            else:
                for table in table_list[:10]:
                    print(f"  • {table}")
                if len(table_list) > 10:
                    print(f"  ... and {len(table_list) - 10} more tables")
            
            # Show training status
            status = self.check_training_status()
            print(f"\n🎓 Training Status:")
            print(f"  • Status: {status['status']}")
            print(f"  • Total training items: {status['training_count']}")
            print(f"  • DDL items: {status['ddl_count']}")
            print(f"  • Documentation items: {status['doc_count']}")
            print(f"  • Q&A pairs: {status['sql_count']}")
            print(f"  • {status['message']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 70)

    # ==================== NEW: AUTO-TRAINING METHOD ====================
    
    def auto_train_on_schema(self, database: str = None, schemas: List[str] = None, 
                             tables: List[str] = None, sample_data: bool = False):
        """
        Automatically train on database schema. This is called ONCE on connection.
        
        Args:
            database (str): Specific database to train on (optional)
            schemas (List[str]): List of schemas to include (optional)
            tables (List[str]): Specific tables to train on (optional)
            sample_data (bool): Whether to include sample data for better context
        """
        if not self.run_sql_is_set:
            raise Exception("Please connect to a database first")
        
        print("🔄 Starting automatic schema training (one-time operation)...")
        start_time = time.time()
        training_count = 0
        
        try:
            # SQLite Implementation
            if self.dialect == "SQLite":
                tables_df = self.run_sql(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                table_names = tables_df['name'].tolist()
                
                # Filter tables if specified
                if tables:
                    table_names = [t for t in table_names if t in tables]
                
                print(f"📊 Found {len(table_names)} tables to process...")
                
                for idx, table in enumerate(table_names, 1):
                    try:
                        # Progress indicator
                        if idx % 5 == 0 or idx == len(table_names):
                            print(f"   Processing table {idx}/{len(table_names)}...")
                        
                        # Get table info
                        table_info = self.run_sql(f"PRAGMA table_info({table})")
                        
                        # Create DDL
                        ddl = f"CREATE TABLE {table} (\n"
                        columns = []
                        for _, row in table_info.iterrows():
                            col_def = f"  {row['name']} {row['type']}"
                            if row['notnull']:
                                col_def += " NOT NULL"
                            if row['pk']:
                                col_def += " PRIMARY KEY"
                            columns.append(col_def)
                        ddl += ",\n".join(columns) + "\n);"
                        
                        # Add DDL to training
                        self.add_ddl(ddl)
                        training_count += 1
                        
                        # Create documentation
                        doc = f"Table: {table}\n\n"
                        doc += f"Description: This table contains {len(table_info)} columns.\n\n"
                        doc += "Columns:\n"
                        for _, row in table_info.iterrows():
                            doc += f"  - {row['name']} ({row['type']})"
                            if row['pk']:
                                doc += " [PRIMARY KEY]"
                            if row['notnull']:
                                doc += " [NOT NULL]"
                            doc += "\n"
                        
                        # Add sample data if requested
                        if sample_data:
                            try:
                                sample_df = self.run_sql(f"SELECT * FROM {table} LIMIT 3")
                                if not sample_df.empty:
                                    doc += f"\nSample data (first 3 rows):\n{sample_df.to_markdown(index=False)}\n"
                            except Exception as e:
                                doc += f"\nNote: Could not fetch sample data - {str(e)}\n"
                        
                        self.add_documentation(doc)
                        training_count += 1
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not process table '{table}': {e}")
                        continue
                
                # Try to get foreign key relationships
                try:
                    fk_doc = "Table Relationships:\n\n"
                    has_relationships = False
                    for table in table_names:
                        try:
                            fk_info = self.run_sql(f"PRAGMA foreign_key_list({table})")
                            if not fk_info.empty:
                                has_relationships = True
                                for _, fk in fk_info.iterrows():
                                    fk_doc += f"  • {table}.{fk['from']} → {fk['table']}.{fk['to']}\n"
                        except:
                            pass
                    
                    if has_relationships:
                        self.add_documentation(fk_doc)
                        training_count += 1
                except:
                    pass
            
            # PostgreSQL Implementation
            elif self.dialect == "PostgreSQL":
                schema_query = """
                    SELECT 
                        table_schema,
                        table_name,
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name, ordinal_position
                """
                
                df_schema = self.run_sql(schema_query)
                
                # Filter by schemas/tables if specified
                if schemas:
                    df_schema = df_schema[df_schema['table_schema'].isin(schemas)]
                if tables:
                    df_schema = df_schema[df_schema['table_name'].isin(tables)]
                
                # Group by table
                grouped = df_schema.groupby(['table_schema', 'table_name'])
                total_tables = len(grouped)
                print(f"📊 Found {total_tables} tables to process...")
                
                for idx, ((schema, table), group) in enumerate(grouped, 1):
                    try:
                        if idx % 5 == 0 or idx == total_tables:
                            print(f"   Processing table {idx}/{total_tables}...")
                        
                        full_table_name = f"{schema}.{table}"
                        
                        # Generate DDL
                        ddl = f"CREATE TABLE {full_table_name} (\n"
                        columns_ddl = []
                        for _, row in group.iterrows():
                            col = f"  {row['column_name']} {row['data_type']}"
                            if row['is_nullable'] == 'NO':
                                col += " NOT NULL"
                            if row['column_default'] is not None:
                                col += f" DEFAULT {row['column_default']}"
                            columns_ddl.append(col)
                        ddl += ",\n".join(columns_ddl) + "\n);"
                        
                        self.add_ddl(ddl)
                        training_count += 1
                        
                        # Generate Documentation
                        doc = f"Table: {full_table_name}\n\n"
                        doc += f"Schema: {schema}\n"
                        doc += f"Description: This table contains {len(group)} columns.\n\n"
                        doc += "Columns:\n"
                        for _, row in group.iterrows():
                            doc += f"  - {row['column_name']} ({row['data_type']})"
                            if row['is_nullable'] == 'NO':
                                doc += " [NOT NULL]"
                            doc += "\n"
                        
                        if sample_data:
                            try:
                                sample_df = self.run_sql(f"SELECT * FROM {full_table_name} LIMIT 3")
                                if not sample_df.empty:
                                    doc += f"\nSample data:\n{sample_df.to_markdown(index=False)}\n"
                            except Exception as e:
                                doc += f"\nNote: Could not fetch sample data - {str(e)}\n"
                        
                        self.add_documentation(doc)
                        training_count += 1
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not process table '{schema}.{table}': {e}")
                        continue
                
                # Try to get foreign key relationships
                try:
                    fk_query = """
                        SELECT
                            tc.table_schema, 
                            tc.table_name, 
                            kcu.column_name,
                            ccu.table_schema AS foreign_table_schema,
                            ccu.table_name AS foreign_table_name,
                            ccu.column_name AS foreign_column_name
                        FROM information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                            ON ccu.constraint_name = tc.constraint_name
                            AND ccu.table_schema = tc.table_schema
                        WHERE tc.constraint_type = 'FOREIGN KEY'
                    """
                    df_fk = self.run_sql(fk_query)
                    if not df_fk.empty:
                        fk_doc = "Table Relationships (Foreign Keys):\n\n"
                        for _, fk in df_fk.iterrows():
                            fk_doc += f"  • {fk['table_schema']}.{fk['table_name']}.{fk['column_name']} → "
                            fk_doc += f"{fk['foreign_table_schema']}.{fk['foreign_table_name']}.{fk['foreign_column_name']}\n"
                        self.add_documentation(fk_doc)
                        training_count += 1
                except:
                    pass
            
            # MySQL Implementation
            elif self.dialect == "MySQL":
                schema_query = """
                    SELECT 
                        table_schema,
                        table_name,
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        column_key
                    FROM information_schema.columns
                    WHERE table_schema = DATABASE()
                    ORDER BY table_name, ordinal_position
                """
                
                df_schema = self.run_sql(schema_query)
                
                if tables:
                    df_schema = df_schema[df_schema['table_name'].isin(tables)]
                
                grouped = df_schema.groupby('table_name')
                total_tables = len(grouped)
                print(f"📊 Found {total_tables} tables to process...")
                
                for idx, (table, group) in enumerate(grouped, 1):
                    try:
                        if idx % 5 == 0 or idx == total_tables:
                            print(f"   Processing table {idx}/{total_tables}...")
                        
                        # Generate DDL
                        ddl = f"CREATE TABLE {table} (\n"
                        columns_ddl = []
                        for _, row in group.iterrows():
                            col = f"  {row['column_name']} {row['data_type']}"
                            if row['is_nullable'] == 'NO':
                                col += " NOT NULL"
                            if row['column_key'] == 'PRI':
                                col += " PRIMARY KEY"
                            columns_ddl.append(col)
                        ddl += ",\n".join(columns_ddl) + "\n);"
                        
                        self.add_ddl(ddl)
                        training_count += 1
                        
                        # Generate Documentation
                        doc = f"Table: {table}\n\n"
                        doc += f"Description: This table contains {len(group)} columns.\n\n"
                        doc += "Columns:\n"
                        for _, row in group.iterrows():
                            doc += f"  - {row['column_name']} ({row['data_type']})"
                            if row['column_key'] == 'PRI':
                                doc += " [PRIMARY KEY]"
                            if row['is_nullable'] == 'NO':
                                doc += " [NOT NULL]"
                            doc += "\n"
                        
                        if sample_data:
                            try:
                                sample_df = self.run_sql(f"SELECT * FROM {table} LIMIT 3")
                                if not sample_df.empty:
                                    doc += f"\nSample data:\n{sample_df.to_markdown(index=False)}\n"
                            except Exception as e:
                                doc += f"\nNote: Could not fetch sample data\n"
                        
                        self.add_documentation(doc)
                        training_count += 1
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not process table '{table}': {e}")
                        continue
            
            # Generic fallback for other databases
            else:
                print(f"⚠️  Generic training for {self.dialect}...")
                table_list = self._get_cached_table_list()
                print(f"📊 Found {len(table_list)} tables")
                
                for idx, table in enumerate(table_list, 1):
                    try:
                        if idx % 5 == 0 or idx == len(table_list):
                            print(f"   Processing table {idx}/{len(table_list)}...")
                        
                        # Try to describe the table
                        try:
                            desc_df = self.run_sql(f"DESCRIBE {table}")
                        except:
                            try:
                                desc_df = self.run_sql(f"SELECT * FROM {table} LIMIT 0")
                                desc_df = pd.DataFrame({'column': desc_df.columns, 'type': desc_df.dtypes.astype(str)})
                            except:
                                continue
                        
                        doc = f"Table: {table}\n\n"
                        doc += f"Columns:\n{desc_df.to_markdown(index=False)}\n"
                        
                        self.add_documentation(doc)
                        training_count += 1
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not process table '{table}': {e}")
                        continue
            
            elapsed = time.time() - start_time
            print(f"✅ Training completed in {elapsed:.2f} seconds!")
            print(f"📚 Added {training_count} training items")
            
            # Mark as trained
            self._trained = True
            
        except Exception as e:
            print(f"❌ Error during auto-training: {e}")
            traceback.print_exc()

    # ==================== ENHANCED SQL GENERATION ====================

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        Enhanced SQL generation with automatic schema context.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
            
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"

        return self.extract_sql(llm_response)

    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Enhanced prompt generation with smart fallback for missing context.
        """
        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
            "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        # NEW: Check if we have context, if not provide table list
        has_context = len(ddl_list) > 0 or len(doc_list) > 0 or len(question_sql_list) > 0
        
        if not has_context:
            # Provide helpful guidance with available tables
            cached_tables = self._get_cached_table_list()
            if cached_tables:
                initial_prompt += f"\n\n📋 Available tables in the database: {', '.join(cached_tables)}\n"
                initial_prompt += "Note: Limited schema information is available. Please infer table structures from the available table names.\n"
            else:
                initial_prompt += "\n\n⚠️  Note: No schema information available. "
                initial_prompt += "Please generate SQL based on common database patterns or ask for table names to be specified.\n"
        
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.dialect}-compliant and executable, and free of syntax errors. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    # ==================== ENHANCED CONNECTION METHODS ====================

    def connect_to_sqlite(self, url: str, check_same_thread: bool = False, 
                         auto_train: bool = True, **kwargs):
        """
        Connect to a SQLite database with optional automatic schema training.
        
        Args:
            url (str): The URL/path of the database to connect to.
            check_same_thread (bool): Allow multi-thread access.
            auto_train (bool): Automatically train on database schema (default: True)
        """
        # Path to save the downloaded database
        path = os.path.basename(urlparse(url).path)

        # Download the database if it doesn't exist
        if not os.path.exists(url):
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            url = path

        # Connect to the database
        conn = sqlite3.connect(url, check_same_thread=check_same_thread, **kwargs)

        def run_sql_sqlite(sql: str):
            return pd.read_sql_query(sql, conn)

        self.dialect = "SQLite"
        self.run_sql = run_sql_sqlite
        self.run_sql_is_set = True
        
        # NEW: Auto-train on connection
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        auto_train: bool = True,
        **kwargs
    ):
        """
        Connect to postgres with automatic schema training.
        
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
            auto_train (bool): Automatically train on database schema (default: True)
        """
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[postgres]"
            )

        if not host:
            host = os.getenv("HOST")
        if not host:
            raise ImproperlyConfigured("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("DATABASE")
        if not dbname:
            raise ImproperlyConfigured("Please set your postgres database")

        if not user:
            user = os.getenv("PG_USER")
        if not user:
            raise ImproperlyConfigured("Please set your postgres user")

        if not password:
            password = os.getenv("PASSWORD")
        if not password:
            raise ImproperlyConfigured("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")
        if not port:
            raise ImproperlyConfigured("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs
            )
        except psycopg2.Error as e:
            raise ValidationError(e)

        def connect_to_db():
            return psycopg2.connect(host=host, dbname=dbname,
                        user=user, password=password, port=port, **kwargs)

        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                if conn:
                    conn.close()
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise ValidationError(e)

            except Exception as e:
                conn.rollback()
                raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres
        
        # NEW: Auto-train on connection
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_mysql(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        auto_train: bool = True,
        **kwargs
    ):
        """
        Connect to MySQL with automatic schema training.
        """
        try:
            import pymysql.cursors
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install PyMySQL"
            )

        if not host:
            host = os.getenv("HOST")
        if not host:
            raise ImproperlyConfigured("Please set your MySQL host")

        if not dbname:
            dbname = os.getenv("DATABASE")
        if not dbname:
            raise ImproperlyConfigured("Please set your MySQL database")

        if not user:
            user = os.getenv("USER")
        if not user:
            raise ImproperlyConfigured("Please set your MySQL user")

        if not password:
            password = os.getenv("PASSWORD")
        if not password:
            raise ImproperlyConfigured("Please set your MySQL password")

        if not port:
            port = os.getenv("PORT")
        if not port:
            raise ImproperlyConfigured("Please set your MySQL port")

        conn = None

        try:
            conn = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=dbname,
                port=port,
                cursorclass=pymysql.cursors.DictCursor,
                **kwargs
            )
        except pymysql.Error as e:
            raise ValidationError(e)

        def run_sql_mysql(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    conn.ping(reconnect=True)
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except pymysql.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_mysql
        self.dialect = "MySQL"
        
        # NEW: Auto-train on connection
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_duckdb(self, url: str, init_sql: str = None, 
                         auto_train: bool = True, **kwargs):
        """
        Connect to a DuckDB database with automatic schema training.
        
        Args:
            url (str): The URL of the database. Use :memory: for in-memory.
            init_sql (str, optional): SQL to run on connection.
            auto_train (bool): Automatically train on database schema (default: True)
        """
        try:
            import duckdb
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[duckdb]"
            )

        if url == ":memory:" or url == "":
            path = ":memory:"
        else:
            print(os.path.exists(url))
            if os.path.exists(url):
                path = url
            elif url.startswith("md") or url.startswith("motherduck"):
                path = url
            else:
                path = os.path.basename(urlparse(url).path)
                if not os.path.exists(path):
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(path, "wb") as f:
                        f.write(response.content)

        conn = duckdb.connect(path, **kwargs)
        if init_sql:
            conn.query(init_sql)

        def run_sql_duckdb(sql: str):
            return conn.query(sql).to_df()

        self.dialect = "DuckDB SQL"
        self.run_sql = run_sql_duckdb
        self.run_sql_is_set = True
        
        # NEW: Auto-train on connection
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    # ==================== KEEP ALL ORIGINAL METHODS ====================
    # The following methods remain unchanged from the original implementation

    def extract_sql(self, llm_response: str) -> str:
        """Extract SQL query from LLM response."""
        import re

        # Match CREATE TABLE ... AS SELECT
        sqls = re.findall(r"\bCREATE\s+TABLE\b.*?\bAS\b.*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match WITH clause (CTEs)
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match SELECT ... ;
        sqls = re.findall(r"\bSELECT\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match ```sql ... ``` blocks
        sqls = re.findall(r"```sql\s*\n(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1].strip()
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match any ``` ... ``` code blocks
        sqls = re.findall(r"```(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1].strip()
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response

    def is_sql_valid(self, sql: str) -> bool:
        """Check if SQL query is valid."""
        parsed = sqlparse.parse(sql)
        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True
        return False

    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        """Check if a chart should be generated."""
        if len(df) > 1 and df.select_dtypes(include=['number']).shape[1] > 0:
            return True
        return False

    def generate_rewritten_question(self, last_question: str, new_question: str, **kwargs) -> str:
        """Generate a rewritten question by combining related questions."""
        if last_question is None:
            return new_question

        prompt = [
            self.system_message("Your goal is to combine a sequence of questions into a singular question if they are related. If the second question does not relate to the first question and is fully self-contained, return the second question. Return just the new combined question with no additional explanations. The question should theoretically be answerable with a single SQL statement."),
            self.user_message("First question: " + last_question + "\nSecond question: " + new_question),
        ]

        return self.submit_prompt(prompt=prompt, **kwargs)

    def generate_followup_questions(
        self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:
        """Generate followup questions."""
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe SQL query for this question was: {sql}\n\nThe following is a pandas DataFrame with the results of the query: \n{df.head(25).to_markdown()}\n\n"
            ),
            self.user_message(
                f"Generate a list of {n_questions} followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions. Remember that there should be an unambiguous SQL query that can be generated from the question. Prefer questions that are answerable outside of the context of this conversation. Prefer questions that are slight modifications of the SQL query that was generated that allow digging deeper into the data. Each question will be turned into a button that the user can click to generate a new SQL query so don't use 'example' type questions. Each question must have a one-to-one correspondence with an instantiated SQL query." +
                self._response_language()
            ),
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)
        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    def generate_questions(self, **kwargs) -> List[str]:
        """Generate a list of questions."""
        question_sql = self.get_similar_question_sql(question="", **kwargs)
        return [q["question"] for q in question_sql]

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        """Generate a summary of query results."""
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)
        return summary

    # ==================== ABSTRACT METHODS (must be implemented by subclasses) ====================
    
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        pass

    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        pass

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        pass

    # ==================== HELPER METHODS ====================

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"
            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"
        return initial_prompt

    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"
            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"
        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"
            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"
        return initial_prompt

    def generate_question(self, sql: str, **kwargs) -> str:
        """Generate a question from SQL."""
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )
        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        """Extract Python code from markdown."""
        markdown_string = markdown_string.strip()
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())
        if len(python_code) == 0:
            return markdown_string
        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        """Remove fig.show() from plotly code."""
        plotly_code = raw_plotly_code.replace("fig.show()", "")
        return plotly_code

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        """Generate Plotly visualization code."""
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)
        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    # ==================== REMAINING CONNECTION METHODS ====================
    # Keep all other connect methods from original (Snowflake, BigQuery, Oracle, etc.)
    # These would be updated similarly with auto_train parameter

    def connect_to_snowflake(
        self,
        account: str,
        username: str,
        password: str,
        database: str,
        role: Union[str, None] = None,
        warehouse: Union[str, None] = None,
        auto_train: bool = True,
        **kwargs
    ):
        """Connect to Snowflake with auto-training."""
        try:
            snowflake = __import__("snowflake.connector")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[snowflake]"
            )

        if username == "my-username":
            username_env = os.getenv("SNOWFLAKE_USERNAME")
            if username_env is not None:
                username = username_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake username.")

        if password == "mypassword":
            password_env = os.getenv("SNOWFLAKE_PASSWORD")
            if password_env is not None:
                password = password_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake password.")

        if account == "my-account":
            account_env = os.getenv("SNOWFLAKE_ACCOUNT")
            if account_env is not None:
                account = account_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake account.")

        if database == "my-database":
            database_env = os.getenv("SNOWFLAKE_DATABASE")
            if database_env is not None:
                database = database_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake database.")

        conn = snowflake.connector.connect(
            user=username,
            password=password,
            account=account,
            database=database,
            client_session_keep_alive=True,
            **kwargs
        )

        def run_sql_snowflake(sql: str) -> pd.DataFrame:
            cs = conn.cursor()
            if role is not None:
                cs.execute(f"USE ROLE {role}")
            if warehouse is not None:
                cs.execute(f"USE WAREHOUSE {warehouse}")
            cs.execute(f"USE DATABASE {database}")
            cur = cs.execute(sql)
            results = cur.fetchall()
            df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])
            return df

        self.dialect = "Snowflake SQL"
        self.run_sql = run_sql_snowflake
        self.run_sql_is_set = True
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_bigquery(
        self,
        cred_file_path: str = None,
        project_id: str = None,
        auto_train: bool = True,
        **kwargs
    ):
        """Connect to BigQuery with auto-training."""
        try:
            from google.api_core.exceptions import GoogleAPIError
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[bigquery]"
            )

        if not project_id:
            project_id = os.getenv("PROJECT_ID")
        if not project_id:
            raise ImproperlyConfigured("Please set your Google Cloud Project ID.")

        import sys
        if "google.colab" in sys.modules:
            try:
                from google.colab import auth
                auth.authenticate_user()
            except Exception as e:
                raise ImproperlyConfigured(e)

        conn = None
        if not cred_file_path:
            try:
                conn = bigquery.Client(project=project_id)
            except:
                print("Could not found any google cloud implicit credentials")
        else:
            validate_config_path(cred_file_path)

        if not conn:
            with open(cred_file_path, "r") as f:
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(f.read()),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            try:
                conn = bigquery.Client(project=project_id, credentials=credentials, **kwargs)
            except:
                raise ImproperlyConfigured("Could not connect to bigquery please correct credentials")

        def run_sql_bigquery(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                job = conn.query(sql)
                df = job.result().to_dataframe()
                return df
            return None

        self.dialect = "BigQuery SQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_bigquery
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    # Continue with other connection methods (Oracle, ClickHouse, MSSQL, Presto, Hive)
    # All follow the same pattern with auto_train parameter

    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """Run a SQL query on the connected database."""
        raise Exception(
            "You need to connect to a database first by running vn.connect_to_snowflake(), vn.connect_to_postgres(), similar function, or manually set vn.run_sql"
        )

    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = True,
        visualize: bool = True,
        allow_llm_to_see_data: bool = False,
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None],
            Union[plotly.graph_objs.Figure, None],
        ],
        None,
    ]:
        """Ask Vanna.AI a question and get the SQL query that answers it."""
        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            print(e)
            return None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
            except Exception as e:
                print(sql)

        if self.run_sql_is_set is False:
            print("If you want to run the SQL query, connect to a database first.")
            if print_results:
                return None
            else:
                return sql, None, None

        try:
            df = self.run_sql(sql)

            if print_results:
                try:
                    display = __import__("IPython.display", fromList=["display"]).display
                    display(df)
                except Exception as e:
                    print(df)

            if len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)

            if visualize:
                try:
                    plotly_code = self.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    if print_results:
                        try:
                            display = __import__("IPython.display", fromlist=["display"]).display
                            Image = __import__("IPython.display", fromlist=["Image"]).Image
                            img_bytes = fig.to_image(format="png", scale=2)
                            display(Image(img_bytes))
                        except Exception as e:
                            fig.show()
                except Exception as e:
                    traceback.print_exc()
                    print("Couldn't run plotly code: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, None, None
        return sql, df, fig

    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        """Train Vanna.AI on a question and its corresponding SQL query."""
        if question and not sql:
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                print("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                print(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES")
        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        """
        Generate a training plan from an information schema dataframe.
        """
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column, schema_column, table_column]
        candidates = ["column_name", "data_type", "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[columns].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan

    def get_training_plan_snowflake(
        self,
        filter_databases: Union[List[str], None] = None,
        filter_schemas: Union[List[str], None] = None,
        include_information_schema: bool = False,
        use_historical_queries: bool = True,
    ) -> TrainingPlan:
        plan = TrainingPlan([])

        if self.run_sql_is_set is False:
            raise ImproperlyConfigured("Please connect to a database first.")

        if use_historical_queries:
            try:
                print("Trying query history")
                df_history = self.run_sql(
                    """ select * from table(information_schema.query_history(result_limit => 5000)) order by start_time"""
                )

                df_history_filtered = df_history.query("ROWS_PRODUCED > 1")
                if filter_databases is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_databases]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if filter_schemas is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_schemas]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if len(df_history_filtered) > 10:
                    df_history_filtered = df_history_filtered.sample(10)

                for query in df_history_filtered["QUERY_TEXT"].unique().tolist():
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_SQL,
                            item_group="",
                            item_name=self.generate_question(query),
                            item_value=query,
                        )
                    )

            except Exception as e:
                print(e)

        databases = self._get_databases()

        for database in databases:
            if filter_databases is not None and database not in filter_databases:
                continue

            try:
                df_tables = self._get_information_schema_tables(database=database)

                print(f"Trying INFORMATION_SCHEMA.COLUMNS for {database}")
                df_columns = self.run_sql(
                    f"SELECT * FROM {database}.INFORMATION_SCHEMA.COLUMNS"
                )

                for schema in df_tables["TABLE_SCHEMA"].unique().tolist():
                    if filter_schemas is not None and schema not in filter_schemas:
                        continue

                    if (
                        not include_information_schema
                        and schema == "INFORMATION_SCHEMA"
                    ):
                        continue

                    df_columns_filtered_to_schema = df_columns.query(
                        f"TABLE_SCHEMA == '{schema}'"
                    )

                    try:
                        tables = (
                            df_columns_filtered_to_schema["TABLE_NAME"]
                            .unique()
                            .tolist()
                        )

                        for table in tables:
                            df_columns_filtered_to_table = (
                                df_columns_filtered_to_schema.query(
                                    f"TABLE_NAME == '{table}'"
                                )
                            )
                            doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                            doc += df_columns_filtered_to_table[
                                [
                                    "TABLE_CATALOG",
                                    "TABLE_SCHEMA",
                                    "TABLE_NAME",
                                    "COLUMN_NAME",
                                    "DATA_TYPE",
                                    "COMMENT",
                                ]
                            ].to_markdown()

                            plan._plan.append(
                                TrainingPlanItem(
                                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                                    item_group=f"{database}.{schema}",
                                    item_name=table,
                                    item_value=doc,
                                )
                            )

                    except Exception as e:
                        print(e)
                        pass
            except Exception as e:
                print(e)

        return plan

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        Get a Plotly figure from a dataframe and Plotly code.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)
            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                fig = px.pie(df, names=categorical_cols[0])
            else:
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig

    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ) -> list:
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    # ==================== REMAINING DB CONNECTION METHODS ====================
    
    def connect_to_clickhouse(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        auto_train: bool = True,
        **kwargs
    ):
        """Connect to ClickHouse with auto-training."""
        try:
            import clickhouse_connect
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install clickhouse_connect"
            )

        if not host:
            host = os.getenv("HOST")
        if not host:
            raise ImproperlyConfigured("Please set your ClickHouse host")

        if not dbname:
            dbname = os.getenv("DATABASE")
        if not dbname:
            raise ImproperlyConfigured("Please set your ClickHouse database")

        if not user:
            user = os.getenv("USER")
        if not user:
            raise ImproperlyConfigured("Please set your ClickHouse user")

        if not password:
            password = os.getenv("PASSWORD")
        if not password:
            raise ImproperlyConfigured("Please set your ClickHouse password")

        if not port:
            port = os.getenv("PORT")
        if not port:
            raise ImproperlyConfigured("Please set your ClickHouse port")

        conn = None

        try:
            conn = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=user,
                password=password,
                database=dbname,
                **kwargs
            )
            print(conn)
        except Exception as e:
            raise ValidationError(e)

        def run_sql_clickhouse(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    result = conn.query(sql)
                    results = result.result_rows
                    df = pd.DataFrame(results, columns=result.column_names)
                    return df
                except Exception as e:
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_clickhouse
        self.dialect = "ClickHouse"
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_oracle(
        self,
        user: str = None,
        password: str = None,
        dsn: str = None,
        auto_train: bool = True,
        **kwargs
    ):
        """Connect to Oracle with auto-training."""
        try:
            import oracledb
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install oracledb"
            )

        if not dsn:
            dsn = os.getenv("DSN")
        if not dsn:
            raise ImproperlyConfigured("Please set your Oracle dsn which should include host:port/sid")

        if not user:
            user = os.getenv("USER")
        if not user:
            raise ImproperlyConfigured("Please set your Oracle db user")

        if not password:
            password = os.getenv("PASSWORD")
        if not password:
            raise ImproperlyConfigured("Please set your Oracle db password")

        conn = None

        try:
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                **kwargs
            )
        except oracledb.Error as e:
            raise ValidationError(e)

        def run_sql_oracle(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(';'):
                        sql = sql[:-1]

                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except oracledb.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_oracle
        self.dialect = "Oracle SQL"
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_mssql(self, odbc_conn_str: str, auto_train: bool = True, **kwargs):
        """Connect to Microsoft SQL Server with auto-training."""
        try:
            import pyodbc
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install pyodbc"
            )

        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install sqlalchemy"
            )

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        from sqlalchemy import create_engine
        engine = create_engine(connection_url, **kwargs)

        def run_sql_mssql(sql: str):
            with engine.begin() as conn:
                df = pd.read_sql_query(sa.text(sql), conn)
                conn.close()
                return df
            raise Exception("Couldn't run sql")

        self.dialect = "T-SQL / Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_presto(
        self,
        host: str,
        catalog: str = 'hive',
        schema: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        combined_pem_path: str = None,
        protocol: str = 'https',
        requests_kwargs: dict = None,
        auto_train: bool = True,
        **kwargs
    ):
        """Connect to Presto with auto-training."""
        try:
            from pyhive import presto
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install pyhive"
            )

        if not host:
            host = os.getenv("PRESTO_HOST")
        if not host:
            raise ImproperlyConfigured("Please set your presto host")

        if not catalog:
            catalog = os.getenv("PRESTO_CATALOG")
        if not catalog:
            raise ImproperlyConfigured("Please set your presto catalog")

        if not user:
            user = os.getenv("PRESTO_USER")
        if not user:
            raise ImproperlyConfigured("Please set your presto user")

        if not password:
            password = os.getenv("PRESTO_PASSWORD")

        if not port:
            port = os.getenv("PRESTO_PORT")
        if not port:
            raise ImproperlyConfigured("Please set your presto port")

        conn = None

        try:
            if requests_kwargs is None and combined_pem_path is not None:
                requests_kwargs = {'verify': combined_pem_path}
            conn = presto.Connection(host=host,
                                     username=user,
                                     password=password,
                                     catalog=catalog,
                                     schema=schema,
                                     port=port,
                                     protocol=protocol,
                                     requests_kwargs=requests_kwargs,
                                     **kwargs)
        except presto.Error as e:
            raise ValidationError(e)

        def run_sql_presto(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(';'):
                        sql = sql[:-1]
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except presto.Error as e:
                    print(e)
                    raise ValidationError(e)

                except Exception as e:
                    print(e)
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_presto
        self.dialect = "Presto SQL"
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True

    def connect_to_hive(
        self,
        host: str = None,
        dbname: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        auth: str = 'CUSTOM',
        auto_train: bool = True,
        **kwargs
    ):
        """Connect to Hive with auto-training."""
        try:
            from pyhive import hive
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install pyhive"
            )

        if not host:
            host = os.getenv("HIVE_HOST")
        if not host:
            raise ImproperlyConfigured("Please set your hive host")

        if not dbname:
            dbname = os.getenv("HIVE_DATABASE")
        if not dbname:
            raise ImproperlyConfigured("Please set your hive database")

        if not user:
            user = os.getenv("HIVE_USER")
        if not user:
            raise ImproperlyConfigured("Please set your hive user")

        if not password:
            password = os.getenv("HIVE_PASSWORD")

        if not port:
            port = os.getenv("HIVE_PORT")
        if not port:
            raise ImproperlyConfigured("Please set your hive port")

        conn = None

        try:
            conn = hive.Connection(host=host,
                                   username=user,
                                   password=password,
                                   database=dbname,
                                   port=port,
                                   auth=auth)
        except hive.Error as e:
            raise ValidationError(e)

        def run_sql_hive(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except hive.Error as e:
                    print(e)
                    raise ValidationError(e)

                except Exception as e:
                    print(e)
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_hive
        self.dialect = "Hive SQL"
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True
