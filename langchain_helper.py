import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import LLMChain, SequentialChain
from api_key import openapi_key
import os

os.environ["OPENAI_API_KEY"] = openapi_key
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for it. Suggest only one name."
)
    name_chain = LLMChain(
    llm=chat_model,
    prompt=prompt_template_name,
    output_key="restaurant_name"
    )

# Step 2: Suggest menu items for the restaurant
    prompt_template_items = PromptTemplate(
    input_variables=['restaurant_name'],
    template="Suggest some menu items for {restaurant_name}. Return it as a comma-separated string."
    )
    food_items_chain = LLMChain(
    llm=chat_model,
    prompt=prompt_template_items,
    output_key="menu_items"
)

# Sequential Chain
    chain = SequentialChain(
    chains=[name_chain, food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name', 'menu_items'],
    verbose=True
)

# Run the chain
    result = chain({'cuisine': cuisine})
    return result

if __name__=="__main__":
    print(generate_restaurant_name_and_items("Bangladeshi"))
  
