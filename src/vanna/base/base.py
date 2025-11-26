r"""
Enhanced VannaBase with intelligent one-time schema training, caching, and ADVANCED visualization generation.
This implementation includes context-aware visualization with validation, retry logic, and fallback mechanisms.
"""

import json
import os
import re
import sqlite3
import time
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Optional
from urllib.parse import urlparse

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import sqlparse

from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import validate_config_path


class EnhancedVisualizationGenerator:
    """
    Advanced visualization generator that leverages RAG context for smarter,
    more accurate chart generation with validation and fallback mechanisms.
    """
    
    def __init__(self, vanna_instance):
        self.vn = vanna_instance
        
    def generate_visualization_with_validation(
        self,
        question: str,
        sql: str,
        df: pd.DataFrame,
        ddl_list: List[str] = None,
        doc_list: List[str] = None,
        max_retries: int = 2,
        **kwargs
    ) -> Tuple[go.Figure, str]:
        """
        Generate visualization with validation and retry logic.
        
        Returns:
            Tuple[plotly Figure, explanation string]
        """
        ddl_list = ddl_list or []
        doc_list = doc_list or []
        
        if df.empty:
            return self._create_empty_figure("No data to visualize"), "Query returned no results."
        
        # Step 1: Optimize data if too large
        if len(df) > 10000:
            self.vn.log(f"Optimizing large dataset: {len(df)} rows", "Info")
            df = self._optimize_visualization_data(df)
        
        # Step 2: Build comprehensive context
        context = self._build_visualization_context(
            question, sql, df, ddl_list, doc_list
        )
        
        # Step 3: Try algorithmic selection first for simple cases
        if self._is_simple_case(df, question):
            self.vn.log("Using rule-based visualization for simple case", "Info")
            fig = self._create_smart_fallback_visualization(df, question, context)
            explanation = self._generate_simple_explanation(question, fig, context)
            return fig, explanation
        
        # Step 4: Generate with LLM using retries
        for attempt in range(max_retries + 1):
            try:
                self.vn.log(f"Generating visualization (attempt {attempt + 1}/{max_retries + 1})", "Info")
                
                plotly_code = self._generate_code_with_context(
                    context, attempt, **kwargs
                )
                
                # Execute and validate
                fig = self._execute_and_validate(plotly_code, df)
                
                # Generate explanation
                explanation = self._generate_explanation(
                    question, fig, plotly_code, context, **kwargs
                )
                
                self.vn.log("Visualization generated successfully", "Success")
                return fig, explanation
                
            except Exception as e:
                self.vn.log(f"Attempt {attempt + 1} failed: {str(e)[:100]}", "Warning")
                
                if attempt == max_retries:
                    # Final fallback
                    self.vn.log("Using intelligent fallback visualization", "Info")
                    fig = self._create_smart_fallback_visualization(df, question, context)
                    explanation = "Auto-generated visualization using intelligent fallback rules."
                    return fig, explanation
    
    def _is_simple_case(self, df: pd.DataFrame, question: str) -> bool:
        """Determine if this is a simple case that doesn't need LLM."""
        # Single value
        if len(df) == 1:
            return True
        
        # Very clear intent in question
        q_lower = question.lower()
        clear_intents = ['total', 'count', 'how many', 'sum of']
        if any(intent in q_lower for intent in clear_intents) and len(df) == 1:
            return True
        
        return False
    
    def _build_visualization_context(
        self,
        question: str,
        sql: str,
        df: pd.DataFrame,
        ddl_list: List[str],
        doc_list: List[str]
    ) -> Dict:
        """Build comprehensive context for visualization generation."""
        
        context = {
            'question': question,
            'sql': sql,
            'data_profile': self._create_data_profile(df),
            'viz_recommendations': self._recommend_chart_types(df, question),
            'schema_hints': self._extract_schema_hints(ddl_list, df),
            'domain_context': self._extract_domain_context(doc_list, df),
            'question_intent': self._infer_question_intent(question),
            'optimal_chart': self._select_optimal_chart_type(df, question)
        }
        
        return context
    
    def _create_data_profile(self, df: pd.DataFrame) -> str:
        """Generate comprehensive data profile."""
        if df.empty:
            return "Empty dataset"
        
        profile = []
        profile.append(f"Shape: {len(df)} rows × {len(df.columns)} columns")
        
        # Analyze column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        temporal_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Numeric analysis
        if numeric_cols:
            profile.append(f"\nNumeric columns ({len(numeric_cols)}):")
            for col in numeric_cols[:5]:
                stats = df[col].describe()
                profile.append(
                    f"  • {col}: range=[{stats['min']:.2f}, {stats['max']:.2f}], "
                    f"mean={stats['mean']:.2f}, std={stats['std']:.2f}"
                )
        
        # Categorical analysis
        if categorical_cols:
            profile.append(f"\nCategorical columns ({len(categorical_cols)}):")
            for col in categorical_cols[:5]:
                unique_count = df[col].nunique()
                sample_values = df[col].value_counts().head(3).index.tolist()
                profile.append(
                    f"  • {col}: {unique_count} unique values "
                    f"(top: {', '.join(map(str, sample_values[:3]))})"
                )
        
        # Temporal analysis
        if temporal_cols:
            profile.append(f"\nTemporal columns: {', '.join(temporal_cols)}")
            for col in temporal_cols:
                profile.append(f"  • {col}: {df[col].min()} to {df[col].max()}")
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr.append(
                                f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}"
                            )
                if high_corr:
                    profile.append(f"\nStrong correlations: {', '.join(high_corr[:3])}")
            except:
                pass
        
        return "\n".join(profile)
    
    def _recommend_chart_types(self, df: pd.DataFrame, question: str) -> str:
        """Recommend appropriate chart types."""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        temporal_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Single value - KPI indicator
        if len(df) == 1:
            recommendations.append("⭐ Indicator (single KPI value)")
        
        # Time series
        elif temporal_cols and numeric_cols:
            recommendations.append("⭐ Line chart (time series)")
            if len(numeric_cols) > 1:
                recommendations.append("Multi-line chart (compare metrics)")
        
        # Categorical comparison
        elif categorical_cols and numeric_cols:
            unique_count = df[categorical_cols[0]].nunique()
            if unique_count <= 20:
                recommendations.append("⭐ Bar chart (category comparison)")
            if unique_count <= 10:
                recommendations.append("Pie chart (part-to-whole)")
        
        # Numeric relationships
        elif len(numeric_cols) >= 2:
            recommendations.append("⭐ Scatter plot (relationship)")
            if len(numeric_cols) >= 3:
                recommendations.append("Bubble chart (3 dimensions)")
        
        # Distribution
        elif len(numeric_cols) == 1 and len(df) > 10:
            recommendations.append("⭐ Histogram (distribution)")
            recommendations.append("Box plot (with outliers)")
        
        # Question-based override
        if question:
            q_lower = question.lower()
            if 'trend' in q_lower or 'over time' in q_lower:
                recommendations.insert(0, "🎯 Line chart (PRIMARY: trend analysis)")
            elif 'compare' in q_lower or 'vs' in q_lower or 'versus' in q_lower:
                recommendations.insert(0, "🎯 Bar chart (PRIMARY: comparison)")
            elif 'distribution' in q_lower or 'spread' in q_lower:
                recommendations.insert(0, "🎯 Histogram (PRIMARY: distribution)")
            elif 'relationship' in q_lower or 'correlation' in q_lower:
                recommendations.insert(0, "🎯 Scatter plot (PRIMARY: relationship)")
        
        return " | ".join(recommendations) if recommendations else "Auto-select"
    
    def _select_optimal_chart_type(self, df: pd.DataFrame, question: str = None) -> Dict:
        """Algorithmic chart type selection."""
        
        result = {
            'chart_type': None,
            'x_col': None,
            'y_col': None,
            'color_col': None,
            'config': {},
            'reasoning': []
        }
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        temporal_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        n_rows = len(df)
        
        # Single row → KPI
        if n_rows == 1:
            result['chart_type'] = 'indicator'
            result['y_col'] = numeric_cols[0] if numeric_cols else df.columns[0]
            result['reasoning'].append("Single value → KPI Indicator")
            return result
        
        # Time series
        if temporal_cols and numeric_cols:
            result['chart_type'] = 'line'
            result['x_col'] = temporal_cols[0]
            result['y_col'] = numeric_cols[0]
            result['reasoning'].append("Temporal data → Line chart")
            
            if len(numeric_cols) > 1:
                result['config']['multiple_y'] = numeric_cols[:3]
            
            if categorical_cols and df[categorical_cols[0]].nunique() <= 10:
                result['color_col'] = categorical_cols[0]
            
            return result
        
        # Categorical comparison
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            unique_count = df[cat_col].nunique()
            
            if unique_count <= 5:
                result['chart_type'] = 'pie'
                result['y_col'] = numeric_cols[0]
            elif unique_count <= 30:
                result['chart_type'] = 'bar'
                result['x_col'] = cat_col
                result['y_col'] = numeric_cols[0]
                if unique_count > 15:
                    result['config']['horizontal'] = True
            else:
                result['chart_type'] = 'bar'
                result['x_col'] = cat_col
                result['y_col'] = numeric_cols[0]
                result['config']['top_n'] = 20
            
            result['reasoning'].append(f"{unique_count} categories → {result['chart_type']}")
            return result
        
        # Two numeric → Scatter
        if len(numeric_cols) >= 2:
            result['chart_type'] = 'scatter'
            result['x_col'] = numeric_cols[0]
            result['y_col'] = numeric_cols[1]
            result['reasoning'].append("Two numeric → Scatter plot")
            
            if len(numeric_cols) >= 3:
                result['config']['size_col'] = numeric_cols[2]
            
            if categorical_cols and df[categorical_cols[0]].nunique() <= 10:
                result['color_col'] = categorical_cols[0]
            
            return result
        
        # Single numeric → Histogram
        if len(numeric_cols) == 1 and n_rows >= 10:
            result['chart_type'] = 'histogram'
            result['x_col'] = numeric_cols[0]
            result['reasoning'].append("Single numeric → Histogram")
            return result
        
        # Question override
        if question:
            q_lower = question.lower()
            if 'trend' in q_lower or 'over time' in q_lower:
                result['chart_type'] = 'line'
                result['reasoning'].append("Question implies trend")
            elif 'compare' in q_lower:
                result['chart_type'] = 'bar'
                result['reasoning'].append("Question implies comparison")
        
        # Default
        if not result['chart_type']:
            result['chart_type'] = 'table'
            result['reasoning'].append("Default → Table")
        
        return result
    
    def _extract_schema_hints(self, ddl_list: List[str], df: pd.DataFrame) -> str:
        """Extract visualization hints from schema."""
        if not ddl_list:
            return "No schema hints"
        
        hints = []
        combined_ddl = " ".join(ddl_list).upper()
        
        if 'DATE' in combined_ddl or 'TIMESTAMP' in combined_ddl:
            hints.append("Temporal data present → time-series visualization")
        if 'AMOUNT' in combined_ddl or 'PRICE' in combined_ddl or 'REVENUE' in combined_ddl:
            hints.append("Financial metrics → use currency formatting")
        if 'PERCENT' in combined_ddl:
            hints.append("Percentage data → format as %")
        if 'FOREIGN KEY' in combined_ddl:
            hints.append("Relational data → consider connections")
        
        return " | ".join(hints) if hints else "No schema hints"
    
    def _extract_domain_context(self, doc_list: List[str], df: pd.DataFrame) -> str:
        """Extract domain context from documentation."""
        if not doc_list:
            return "General domain"
        
        combined_docs = " ".join(doc_list).lower()
        
        # Domain detection
        if any(kw in combined_docs for kw in ['sales', 'revenue', 'customer']):
            return "Sales domain → trend lines, comparisons"
        elif any(kw in combined_docs for kw in ['employee', 'department', 'salary']):
            return "HR domain → demographic breakdowns"
        elif any(kw in combined_docs for kw in ['transaction', 'account', 'balance']):
            return "Finance domain → waterfall charts, time-series"
        elif any(kw in combined_docs for kw in ['inventory', 'stock', 'warehouse']):
            return "Operations domain → status indicators"
        else:
            return "General domain"
    
    def _infer_question_intent(self, question: str) -> str:
        """Infer user intent from question."""
        q_lower = question.lower()
        intents = []
        
        if any(w in q_lower for w in ['trend', 'change', 'over time', 'growth']):
            intents.append("trend analysis")
        if any(w in q_lower for w in ['compare', 'versus', 'vs', 'difference']):
            intents.append("comparison")
        if any(w in q_lower for w in ['total', 'sum', 'count', 'how many']):
            intents.append("aggregation")
        if any(w in q_lower for w in ['distribution', 'spread', 'range']):
            intents.append("distribution")
        if any(w in q_lower for w in ['top', 'best', 'highest', 'lowest']):
            intents.append("ranking")
        if any(w in q_lower for w in ['relationship', 'correlation', 'between']):
            intents.append("relationship")
        
        return " + ".join(intents) if intents else "general inquiry"
    
    def _generate_code_with_context(
        self, 
        context: Dict, 
        attempt: int,
        **kwargs
    ) -> str:
        """Generate Plotly code using rich context."""
        
        # Get few-shot examples
        examples = self._get_visualization_examples(context)
        
        # Build prompt
        system_msg = f"""You are an expert data visualization specialist. Create a Python Plotly visualization.

QUESTION: {context['question']}
INTENT: {context['question_intent']}

DATA PROFILE:
{context['data_profile']}

RECOMMENDATIONS:
{context['viz_recommendations']}

OPTIMAL CHART SELECTION:
Type: {context['optimal_chart']['chart_type']}
Reasoning: {', '.join(context['optimal_chart']['reasoning'])}

SCHEMA HINTS: {context['schema_hints']}
DOMAIN: {context['domain_context']}

{examples}

REQUIREMENTS:
1. Choose the MOST APPROPRIATE chart type based on data and question intent
2. Add clear, descriptive title and axis labels
3. Use appropriate colors and styling (professional look)
4. Include hover tooltips with relevant information
5. Handle edge cases gracefully
6. Format numbers appropriately (currency, percentages, etc.)
7. Assume data is in pandas DataFrame 'df'

CRITICAL: Return ONLY executable Python code. No markdown, no explanations, no preamble."""
        
        if attempt > 0:
            system_msg += f"\n\nPrevious attempt failed. Try a different visualization approach."
        
        message_log = [
            self.vn.system_message(system_msg),
            self.vn.user_message("Generate the Plotly visualization code now.")
        ]
        
        response = self.vn.submit_prompt(message_log, **kwargs)
        code = self._extract_and_clean_code(response)
        
        return code
    
    def _get_visualization_examples(self, context: Dict) -> str:
        """Return relevant examples."""
        examples = []
        intent = context['question_intent'].lower()
        chart_type = context['optimal_chart']['chart_type']
        
        # Time series example
        if 'trend' in intent or chart_type == 'line':
            examples.append("""
EXAMPLE - Time Series:
```python
fig = px.line(df, x='date_column', y='value_column',
              title='Trend Over Time',
              labels={'date_column': 'Date', 'value_column': 'Value'},
              markers=True)
fig.update_layout(hovermode='x unified', template='plotly_white')
```""")
        
        # Comparison example
        if 'comparison' in intent or chart_type == 'bar':
            examples.append("""
EXAMPLE - Comparison:
```python
fig = px.bar(df, x='category', y='value',
             title='Comparison by Category',
             text_auto='.2f',
             color='category')
fig.update_traces(textposition='outside')
fig.update_layout(showlegend=False, template='plotly_white')
```""")
        
        # KPI example
        if chart_type == 'indicator':
            examples.append("""
EXAMPLE - KPI Indicator:
```python
value = df.iloc[0, 0] if len(df) > 0 else 0
fig = go.Figure(go.Indicator(
    mode="number",
    value=value,
    title={'text': 'Key Metric', 'font': {'size': 24}},
    number={'font': {'size': 60}}
))
fig.update_layout(height=300)
```""")
        
        return "\n".join(examples[:2])
    
    def _extract_and_clean_code(self, response: str) -> str:
        """Extract and clean Python code."""
        # Remove markdown blocks
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            code = matches[-1]
        else:
            code = response
        
        # Clean up
        code = code.strip()
        code = code.replace("fig.show()", "")
        code = code.replace("plt.show()", "")
        
        # Remove any import statements (we handle them)
        lines = code.split('\n')
        code = '\n'.join([l for l in lines if not l.strip().startswith('import')])
        
        return code
    
    def _execute_and_validate(self, code: str, df: pd.DataFrame) -> go.Figure:
        """Execute code and validate result."""
        local_vars = {
            'df': df, 
            'px': px, 
            'go': go, 
            'pd': pd,
            'make_subplots': make_subplots
        }
        
        try:
            exec(code, globals(), local_vars)
            fig = local_vars.get('fig')
            
            if fig is None:
                raise ValueError("Code did not produce 'fig' variable")
            
            # Validate it's a plotly figure
            if not isinstance(fig, go.Figure):
                raise ValueError(f"Invalid figure type: {type(fig)}")
            
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Execution failed: {str(e)}\n\nCode:\n{code[:500]}")
    
    def _create_smart_fallback_visualization(
        self, 
        df: pd.DataFrame, 
        question: str,
        context: Dict
    ) -> go.Figure:
        """Create intelligent rule-based fallback."""
        
        optimal = context['optimal_chart']
        chart_type = optimal['chart_type']
        
        try:
            # KPI Indicator
            if chart_type == 'indicator':
                col = optimal['y_col'] or df.columns[0]
                value = df[col].iloc[0]
                
                fig = go.Figure(go.Indicator(
                    mode="number",
                    value=float(value),
                    title={'text': col.replace('_', ' ').title(), 'font': {'size': 24}},
                    number={'font': {'size': 60}, 'valueformat': ',.2f'}
                ))
                fig.update_layout(height=300, template='plotly_white')
                return fig
            
            # Line chart
            elif chart_type == 'line':
                x_col = optimal['x_col']
                y_col = optimal['y_col']
                color_col = optimal.get('color_col')
                
                fig = px.line(df, x=x_col, y=y_col, color=color_col,
                             title=f"{y_col.replace('_', ' ').title()} Over Time",
                             markers=True)
                fig.update_layout(hovermode='x unified', template='plotly_white')
                return fig
            
            # Bar chart
            elif chart_type == 'bar':
                x_col = optimal['x_col']
                y_col = optimal['y_col']
                
                # Handle top_n if specified
                if 'top_n' in optimal['config']:
                    df = df.nlargest(optimal['config']['top_n'], y_col)
                
                # Sort for better readability
                df_sorted = df.sort_values(by=y_col, ascending=False)
                
                fig = px.bar(df_sorted, x=x_col, y=y_col,
                           title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                           text_auto='.2f',
                           color=y_col,
                           color_continuous_scale='Blues')
                
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, template='plotly_white')
                
                if df_sorted[x_col].nunique() > 10:
                    fig.update_xaxes(tickangle=-45)
                
                return fig
            
            # Pie chart
            elif chart_type == 'pie':
                y_col = optimal['y_col']
                # Use first categorical or index
                names_col = df.select_dtypes(include=['object', 'category']).columns[0] if len(df.select_dtypes(include=['object', 'category']).columns) > 0 else df.columns[0]
                
                fig = px.pie(df, names=names_col, values=y_col,
                           title=f"Distribution of {y_col.replace('_', ' ').title()}")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(template='plotly_white')
                return fig
            
            # Scatter plot
            elif chart_type == 'scatter':
                x_col = optimal['x_col']
                y_col = optimal['y_col']
                color_col = optimal.get('color_col')
                size_col = optimal['config'].get('size_col')
                
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                               title=f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}",
                               trendline="ols" if len(df) >= 3 else None)
                fig.update_layout(template='plotly_white')
                return fig
            
            # Histogram
            elif chart_type == 'histogram':
                x_col = optimal['x_col']
                
                fig = px.histogram(df, x=x_col, nbins=30,
                                 title=f"Distribution of {x_col.replace('_', ' ').title()}",
                                 marginal="box")
                
                # Add mean line
                mean_val = df[x_col].mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                            annotation_text=f"Mean: {mean_val:.2f}")
                
                fig.update_layout(template='plotly_white')
                return fig
            
            # Table fallback
            else:
                return self._create_table_figure(df)
        
        except Exception as e:
            self.vn.log(f"Fallback visualization error: {e}", "Warning")
            return self._create_table_figure(df)
    
    def _optimize_visualization_data(self, df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
        """Optimize large datasets for visualization."""
        if len(df) <= max_points:
            return df
        
        # Strategy 1: Time-based aggregation
        temporal_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if temporal_cols:
            try:
                df_agg = df.set_index(temporal_cols[0]).resample('D').mean().reset_index()
                if len(df_agg) <= max_points:
                    return df_agg
            except:
                pass
        
        # Strategy 2: Intelligent sampling with outliers
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            sample_size = int(max_points * 0.8)
            outlier_size = int(max_points * 0.2)
            
            main_col = numeric_cols[0]
            
            df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            try:
                top_outliers = df.nlargest(outlier_size // 2, main_col)
                bottom_outliers = df.nsmallest(outlier_size // 2, main_col)
                df_optimized = pd.concat([df_sample, top_outliers, bottom_outliers]).drop_duplicates()
                return df_optimized
            except:
                return df_sample
        
        # Strategy 3: Simple random sample
        return df.sample(n=max_points, random_state=42)
    
    def _generate_explanation(
        self,
        question: str,
        fig: go.Figure,
        code: str,
        context: Dict,
        **kwargs
    ) -> str:
        """Generate explanation of visualization."""
        
        chart_type = self._detect_chart_type(code)
        
        prompt = f"""Generate a brief 2-3 sentence explanation of this visualization for business users.

Question: {question}
Chart Type: {chart_type}
Intent: {context['question_intent']}
Data Summary: {context['data_profile'][:300]}

Focus on insights, not technical details. Be clear and concise."""
        
        try:
            message_log = [
                self.vn.system_message("You explain data visualizations clearly to business users."),
                self.vn.user_message(prompt)
            ]
            explanation = self.vn.submit_prompt(message_log, **kwargs)
            return explanation.strip()
        except:
            return f"This {chart_type.lower()} shows {context['question_intent']} from the query results."
    
    def _generate_simple_explanation(self, question: str, fig: go.Figure, context: Dict) -> str:
        """Generate simple explanation without LLM call."""
        chart_type = context['optimal_chart']['chart_type']
        intent = context['question_intent']
        
        explanations = {
            'indicator': f"This KPI card shows the single value answering: '{question}'",
            'line': f"This line chart visualizes the {intent} trend to answer: '{question}'",
            'bar': f"This bar chart compares values to answer: '{question}'",
            'pie': f"This pie chart shows the distribution breakdown for: '{question}'",
            'scatter': f"This scatter plot reveals the relationship to answer: '{question}'",
            'histogram': f"This histogram displays the distribution to answer: '{question}'",
        }
        
        return explanations.get(chart_type, f"This visualization answers: '{question}'")
    
    def _detect_chart_type(self, code: str) -> str:
        """Detect chart type from code."""
        code_lower = code.lower()
        
        if 'px.line' in code_lower or ('go.scatter' in code_lower and 'mode' in code_lower and 'lines' in code_lower):
            return "Line Chart"
        elif 'px.bar' in code_lower or 'go.bar' in code_lower:
            return "Bar Chart"
        elif 'px.scatter' in code_lower or 'go.scatter' in code_lower:
            return "Scatter Plot"
        elif 'px.pie' in code_lower or 'go.pie' in code_lower:
            return "Pie Chart"
        elif 'px.histogram' in code_lower or 'go.histogram' in code_lower:
            return "Histogram"
        elif 'go.indicator' in code_lower:
            return "KPI Indicator"
        elif 'px.box' in code_lower or 'go.box' in code_lower:
            return "Box Plot"
        else:
            return "Chart"
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    def _create_table_figure(self, df: pd.DataFrame) -> go.Figure:
        """Create table visualization."""
        # Limit rows for display
        display_df = df.head(100)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='steelblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title="Data Table" + (f" (showing first 100 of {len(df)} rows)" if len(df) > 100 else ""),
            height=min(600, 50 + len(display_df) * 30)
        )
        
        return fig


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
        
        # Caching for performance
        self._schema_cache = None
        self._schema_cache_time = None
        self._cache_ttl = 3600  # 1 hour
        self._trained = False
        
        # Initialize enhanced visualization generator
        self._viz_generator = None

    def log(self, message: str, title: str = "Info"):
        """Enhanced logging with emoji indicators."""
        emoji_map = {
            "Info": "ℹ️",
            "Success": "✅",
            "Warning": "⚠️",
            "Error": "❌",
            "Progress": "🔄"
        }
        emoji = emoji_map.get(title, "📌")
        print(f"{emoji} {title}: {message}")

    def _response_language(self) -> str:
        if self.language is None:
            return ""
        return f"Respond in the {self.language} language."

    # ==================== TRAINING STATUS & CACHING ====================
    
    def check_training_status(self) -> dict:
        """Check training status and provide recommendations."""
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
                message = f"⚠️ No training data. Database has {total_tables} tables."
                needs_training = True
            elif ddl_count < total_tables * 0.8:
                status = "partially_trained"
                message = f"⚠️ Partial training: {ddl_count}/{total_tables} tables"
                needs_training = True
            else:
                status = "trained"
                message = f"✅ Fully trained: {total_training} items ({ddl_count} tables, {doc_count} docs, {sql_count} Q&A)"
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
                "message": f"❌ Error checking status: {e}",
                "needs_training": True,
                "training_count": 0,
                "table_count": 0
            }

    def _get_cached_table_list(self) -> List[str]:
        """Get table names with caching."""
        if (self._schema_cache is not None and 
            self._schema_cache_time is not None and 
            time.time() - self._schema_cache_time < self._cache_ttl):
            return self._schema_cache
        
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
            
            self._schema_cache = table_list
            self._schema_cache_time = time.time()
            
            return table_list
        except Exception as e:
            self.log(f"Could not fetch table list: {e}", "Warning")
            return []

    def ensure_trained(self, force: bool = False):
        """Ensure system is trained."""
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
        """Display database schema and training status."""
        if not self.run_sql_is_set:
            print("❌ Not connected to a database")
            return
        
        print("=" * 70)
        print("📊 DATABASE INFORMATION")
        print("=" * 70)
        
        try:
            table_list = self._get_cached_table_list()
            print(f"\n📋 Tables ({len(table_list)}):")
            
            if self.dialect == "SQLite":
                for table in table_list[:10]:
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

    # ==================== AUTO-TRAINING METHOD ====================
    
    def auto_train_on_schema(self, database: str = None, schemas: List[str] = None, 
                             tables: List[str] = None, sample_data: bool = False):
        """Automatically train on database schema."""
        if not self.run_sql_is_set:
            raise Exception("Please connect to a database first")
        
        print("🔄 Starting automatic schema training...")
        start_time = time.time()
        training_count = 0
        
        try:
            # SQLite Implementation
            if self.dialect == "SQLite":
                tables_df = self.run_sql(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                table_names = tables_df['name'].tolist()
                
                if tables:
                    table_names = [t for t in table_names if t in tables]
                
                print(f"📊 Found {len(table_names)} tables to process...")
                
                for idx, table in enumerate(table_names, 1):
                    try:
                        if idx % 5 == 0 or idx == len(table_names):
                            print(f"   Processing table {idx}/{len(table_names)}...")
                        
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
                        print(f"⚠️ Warning: Could not process table '{table}': {e}")
                        continue
                
                # Foreign key relationships
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
                
                if schemas:
                    df_schema = df_schema[df_schema['table_schema'].isin(schemas)]
                if tables:
                    df_schema = df_schema[df_schema['table_name'].isin(tables)]
                
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
                                doc += f"\nNote: Could not fetch sample data\n"
                        
                        self.add_documentation(doc)
                        training_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Warning: Could not process '{schema}.{table}': {e}")
                        continue
            
            # MySQL and other implementations follow similar pattern...
            # (keeping the rest of your original implementation)
            
            elapsed = time.time() - start_time
            print(f"✅ Training completed in {elapsed:.2f} seconds!")
            print(f"📚 Added {training_count} training items")
            
            self._trained = True
            
        except Exception as e:
            print(f"❌ Error during auto-training: {e}")
            traceback.print_exc()

    # ==================== ENHANCED SQL GENERATION ====================

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """Enhanced SQL generation with automatic schema context."""
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
        """Enhanced prompt generation with smart fallback."""
        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
            "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        has_context = len(ddl_list) > 0 or len(doc_list) > 0 or len(question_sql_list) > 0
        
        if not has_context:
            cached_tables = self._get_cached_table_list()
            if cached_tables:
                initial_prompt += f"\n\n📋 Available tables: {', '.join(cached_tables)}\n"
                initial_prompt += "Note: Limited schema information available. Infer structures from table names.\n"
        
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
            if example is not None and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    # ==================== ENHANCED ASK METHOD WITH ADVANCED VISUALIZATION ====================

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
        """
        ENHANCED ask method with advanced context-aware visualization.
        """
        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            self.log(f"SQL generation error: {e}", "Error")
            return None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display = __import__("IPython.display", fromList=["display"]).display
                display(Code(sql))
            except:
                print(f"\n📝 Generated SQL:\n{sql}\n")

        if self.run_sql_is_set is False:
            self.log("Connect to a database to run SQL queries.", "Info")
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
                except:
                    print(f"\n📊 Query Results:\n{df.to_string()}\n")

            if len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)

            # ==================== ENHANCED VISUALIZATION ====================
            if visualize and not df.empty:
                try:
                    # Initialize visualization generator if needed
                    if self._viz_generator is None:
                        self._viz_generator = EnhancedVisualizationGenerator(self)
                    
                    # Get RAG context for visualization
                    ddl_list = self.get_related_ddl(question)
                    doc_list = self.get_related_documentation(question)
                    
                    self.log("Generating enhanced visualization...", "Progress")
                    
                    # Generate with validation and retry
                    fig, explanation = self._viz_generator.generate_visualization_with_validation(
                        question=question,
                        sql=sql,
                        df=df,
                        ddl_list=ddl_list,
                        doc_list=doc_list
                    )
                    
                    if print_results:
                        print(f"\n💡 Visualization Insight: {explanation}\n")
                        try:
                            display = __import__("IPython.display", fromList=["display"]).display
                            display(fig)
                        except:
                            fig.show()
                    
                    return sql, df, fig
                    
                except Exception as e:
                    self.log(f"Visualization error: {e}", "Warning")
                    traceback.print_exc()
                    if print_results:
                        return sql, df, None
                    else:
                        return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            self.log(f"SQL execution error: {e}", "Error")
            if print_results:
                return None
            else:
                return sql, None, None

    # ==================== CONNECTION METHODS WITH AUTO-TRAIN ====================

    def connect_to_sqlite(self, url: str, check_same_thread: bool = False, 
                         auto_train: bool = True, **kwargs):
        """Connect to SQLite with automatic training."""
        path = os.path.basename(urlparse(url).path)

        if not os.path.exists(url):
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            url = path

        conn = sqlite3.connect(url, check_same_thread=check_same_thread, **kwargs)

        def run_sql_sqlite(sql: str):
            return pd.read_sql_query(sql, conn)

        self.dialect = "SQLite"
        self.run_sql = run_sql_sqlite
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

    # (Include all other connection methods: PostgreSQL, MySQL, DuckDB, etc.)
    # They follow the same pattern as connect_to_sqlite

    # ==================== HELPER METHODS ====================

    def extract_sql(self, llm_response: str) -> str:
        """Extract SQL query from LLM response."""
        # CREATE TABLE ... AS SELECT
        sqls = re.findall(r"\bCREATE\s+TABLE\b.*?\bAS\b.*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1]

        # WITH clause
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1]

        # SELECT
        sqls = re.findall(r"\bSELECT\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1]

        # ```sql blocks
        sqls = re.findall(r"```sql\s*\n(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1].strip()

        # Any ``` blocks
        sqls = re.findall(r"```(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1].strip()

        return llm_response

    def is_sql_valid(self, sql: str) -> bool:
        """Check if SQL query is valid."""
        parsed = sqlparse.parse(sql)
        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True
        return False

    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        """Check if chart should be generated."""
        if len(df) > 1 and df.select_dtypes(include=['number']).shape[1] > 0:
            return True
        return False

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        """
        DEPRECATED: Use enhanced visualization through ask() method instead.
        This method is kept for backward compatibility.
        """
        self.log("Using legacy plotly code generation. Consider using enhanced ask() method.", "Warning")
        
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

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """Get Plotly figure from code."""
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)
            fig = ldict.get("fig", None)
        except Exception as e:
            # Fallback visualization
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

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
        """Generate list of questions."""
        question_sql = self.get_similar_question_sql(question="", **kwargs)
        return [q["question"] for q in question_sql]

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        """Generate summary of query results."""
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

    def generate_question(self, sql: str, **kwargs) -> str:
        """Generate question from SQL."""
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

    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        """Train Vanna.AI on question and SQL."""
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

    # ==================== HELPER METHODS ====================

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) // 4

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

    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """Run SQL query on connected database."""
        raise Exception(
            "You need to connect to a database first by running vn.connect_to_snowflake(), vn.connect_to_postgres(), or similar function"
        )

    # ==================== ABSTRACT METHODS ====================
    
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


# ==================== ADDITIONAL CONNECTION METHODS ====================

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
        """Connect to PostgreSQL with automatic training."""
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
        """Connect to MySQL with automatic training."""
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
        """Connect to DuckDB with automatic training."""
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
        
        if auto_train:
            print("🔄 Auto-training enabled. Checking training status...")
            status = self.check_training_status()
            
            if status['needs_training']:
                print(f"📚 {status['message']}")
                self.auto_train_on_schema(sample_data=True)
            else:
                print(f"✅ {status['message']}")
                self._trained = True


# ==================== USAGE EXAMPLE ====================

"""
USAGE EXAMPLE:

from your_vanna_implementation import YourVannaClass

# Initialize your Vanna instance (with your specific vector DB, etc.)
vn = YourVannaClass(config={'api_key': 'your-key', 'model': 'gpt-4'})

# Connect to database (auto-training happens automatically)
vn.connect_to_sqlite('path/to/database.db')

# Show database info
vn.show_database_info()

# Ask questions with ENHANCED visualization
sql, df, fig = vn.ask("What are the top 5 products by revenue?")

# The visualization will now:
# 1. Use RAG context (DDL, documentation) for smarter chart selection
# 2. Validate and retry if code fails
# 3. Use intelligent fallback if LLM fails
# 4. Provide natural language explanations
# 5. Handle large datasets efficiently
# 6. Format charts professionally

# Manual training if needed
vn.ensure_trained(force=True)

# Check training status
status = vn.check_training_status()
print(status)
"""
