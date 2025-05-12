import os
from dotenv import load_dotenv



load_dotenv()
oe.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")