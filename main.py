import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import ScrapeWebsiteTool	

# ========== Configurações ==========
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Override de variáveis de ambiente
os.environ.pop("OPENAI_API_KEY", None)
os.environ["CREWAI_LLM_PROVIDER"] = "ollama"
os.environ["CREWAI_EMBEDDINGS_PROVIDER"] = "ollama"
os.environ["CREW_DISABLE_TELEMETRY"] = "1"

# Inicialização do LLM
llm = LLM(
    model="ollama/mistral:7b",
    base_url="http://localhost:11434",
    stream=False,
    api_key="",
    timeout=240
)

# Ajuste de parâmetros do LLM
llm.max_completion_tokens = 1024
llm.temperature = 0.7

# ========== Funções auxiliares ==========
def fetch_price_history(symbol: str, period: str) -> pd.DataFrame:
    """
    Busca o histórico de preços de um ativo via yfinance.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError(f"Não foi possível obter dados para '{symbol}'.")
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de indicadores técnicos no DataFrame:
      - MM50 e MM200
      - RSI14
      - Bandas de Bollinger
    """
    df["MM50"] = df["Close"].rolling(window=50).mean()
    df["MM200"] = df["Close"].rolling(window=200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    rolling20 = df["Close"].rolling(window=20)
    df["BB_middle"] = rolling20.mean()
    df["BB_std"] = rolling20.std()
    df["BB_upper"] = df["BB_middle"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_middle"] - 2 * df["BB_std"]

    return df

# ========== Tool para o CrewAI ==========
@tool
def view_acao(moeda: str, periodo: str) -> str:
    """
    Puxa o histórico de preços de um ativo via yfinance e calcula indicadores:
      - MM50 e MM200
      - RSI14
      - Bandas de Bollinger

    Parâmetros:
    - moeda: símbolo da ação/criptomoeda (ex: "BTC-USD")
    - periodo: período do histórico (ex: "1y")

    Retorna:
    - String formatada com últimos dados.
    """
    try:
        df = fetch_price_history(moeda, periodo)
        df = compute_indicators(df)
    except Exception as e:
        return f"❌ Erro: {e}"

    last_close = df["Close"].iloc[-1]
    last_mm50 = df["MM50"].iloc[-1]
    last_mm200 = df["MM200"].iloc[-1]
    pct_change = (last_close / df["Close"].iloc[0] - 1) * 100

    return (
        f"📊 Dados para {moeda} em {periodo}:\n"
        f"- Último fechamento: {last_close:.2f}\n"
        f"- MM50: {last_mm50:.2f} | MM200: {last_mm200:.2f}\n"
        f"- Variação no período: {pct_change:.2f}%\n"
    )

# ========== Configuração do Agente ==========
busca_noticia = ScrapeWebsiteTool()

especialista_trading = Agent(
    role="Especialista em Análise de Criptomoedas",
    goal=(
        "Fornecer análise de BTC-USD com indicadores técnicos "
        "e gerar recomendações."
    ),
    backstory=(
        "15 anos de experiência em análise de ativos digitais, "
        "otimizando riscos e retornos."
    ),
    verbose=True,
    memory=True,
    allow_delegation=True,
    llm=llm,
    tools=[view_acao],
)

# ========== Tarefa ==========
analise_trading = Task(
    description=(
        "1. Use view_acao para BTC-USD 1y.\n"
        "2. Recalcule MM50, MM200, RSI14, Bollinger.\n"
        "3. Gere gráfico de Close com MM50/MM200.\n"
        "4. Interprete tendências e sinais técnicos.\n"
        "5. Resuma em 3-5 frases e recomende buy/hold/sell."
    ),
    expected_output=(
        "1. Resumo executivo\n"
        "2. Principais indicadores\n"
        "3. Análise de tendência\n"
        "4. Recomendação final"
    ),
    agent=especialista_trading,
    output_file="analise_btc.txt",
)

especialista_noticia = Agent(
    role="Especialista em buscar noticias sobre bitcoin",
    goal="Analisar noticias que encontrar pela web sobre o bitcoin, seja ela positiva ou negativa perante sua analise. Isso ajudará na tomada de decisão",
    backstory="""
    Você parece que nasceu para isso, para buscar noticias sobre o bitcoin e saber como essa noticia vai influenciar na descida ou subida do mesmo.
    """,
    verbose=True,
    memory=True,
    allow_delegation=False,
    tools=[busca_noticia],
    llm=llm
)

procurando_noticias = Task(
    description="""
    Sua tarefa é realizar uma busca aprofundada sobre as notícias mais recentes relacionadas ao Bitcoin, utilizando a ferramenta de scraping disponível.

    Etapas:

    1. Pesquise em fontes confiáveis de notícias financeiras e tecnológicas (ex: CoinDesk, CoinTelegraph, Bloomberg Crypto, etc.) buscando as manchetes e conteúdos publicados nos últimos 7 dias.
    2. Para cada notícia encontrada:
    - Resuma em 2 a 3 frases o conteúdo principal.
    - Classifique a notícia como **POSITIVA**, **NEGATIVA** ou **NEUTRA** em relação ao impacto no preço do Bitcoin.
    - Justifique brevemente sua classificação, mencionando os pontos principais.
    3. Ao final, elabore um relatório consolidado contendo:
    - Lista das 5 notícias mais relevantes, com título, resumo, classificação e link da fonte.
    - Uma breve análise final sobre o sentimento predominante (positivo, negativo ou neutro) e possíveis impactos futuros no mercado de Bitcoin.

    Use linguagem objetiva, profissional e evite termos subjetivos ou sem embasamento. Foque na clareza, precisão e relevância das informações.
    """,
    expected_output=f"""
    1. Lista das 5 notícias mais relevantes com resumo, classificação e link.
    2. Análise final sobre o sentimento predominante e possíveis impactos.
    3. Retorne o que foi achado por você para o {especialista_trading} para auxiliar em sua analise. 
    """,
    agent=especialista_noticia
)

especialista_decisao = Agent(
    role="Especialista em tomada de decisão sobre Bitcoin",
    goal="Integrar a analise técnica com a analise de noticias para chegar em uma conclusão.",
    backstory="""
    Você é um analista sênior de investimentos, especializado em criptomoedas, responsável por consolidar informações técnicas e de mercado para orientar decisões financeiras estratégicas.
    Seu foco é sempre ser objetivo, preciso e fundamentado em dados.
    """,
    memory=True,
    verbose=True,
    llm=llm,
    allow_delegation=False
)

tomada_decisao = Task(
    description="""
    Sua tarefa é consolidar as análises geradas pelos outros especialistas e formular uma recomendação estratégica sobre o Bitcoin.

    Você receberá:

    1. A análise técnica do agente 'Especialista em Trading', contendo indicadores técnicos, tendências e uma sugestão preliminar de buy/hold/sell.
    2. O relatório de notícias do agente 'Especialista em buscar notícias', com a classificação de sentimento do mercado e possíveis impactos futuros.

    Com base nesses dois relatórios, siga estas etapas:

    1. Resuma brevemente os principais pontos de cada relatório (máximo de 3 frases cada).
    2. Identifique se há convergência ou divergência entre a análise técnica e o sentimento do mercado.
    3. Forneça uma recomendação final: **COMPRAR**, **MANTER** ou **VENDER**, justificando sua escolha com base nos dados integrados.
    4. Indique possíveis riscos e pontos de atenção a serem monitorados nas próximas semanas.

    Utilize uma linguagem clara, objetiva e profissional.
    """,
    expected_output="""
    1. Resumo da análise técnica.
    2. Resumo da análise de notícias.
    3. Diagnóstico: convergência ou divergência nas análises.
    4. Recomendação final: COMPRAR, MANTER ou VENDER, com justificativa.
    5. Lista de riscos e pontos de atenção.
    """,
    agent=especialista_decisao,
    output_file="recomendacao_final_btc.txt"
)

# ========== Execução ==========
def main():
    crew = Crew(
        agents=[especialista_trading, especialista_noticia, especialista_decisao],
        tasks=[analise_trading, procurando_noticias, tomada_decisao],
        verbose=True,
        process=Process.sequential,
    )
    resultado = crew.kickoff()
    logging.info("=== ANÁLISE FINAL ===")
    print(resultado)

if __name__ == "__main__":
    main()
