import scrapy

class EpicuriousRecipeSpider(scrapy.Spider):
    name = "epicurious_recipe"

    # URL of the recipe page
    start_urls = [r"C:\Users\tlebr\Downloads\Polish Lazanki Recipe _ Epicurious.html"]

    def parse(self, response):
        # Extracting the title of the recipe
        title = response.css('h1[data-testid="ContentHeaderHed"]::text').get()

        # Extracting ingredients from different potential sections
        # Option 1: Ingredients could be listed under `div[data-testid="IngredientList"] ul li`
        ingredients = response.css('div[data-testid="IngredientList"] ul li::text').getall()

        # Extracting instructions
        instructions = response.css('div[data-testid="InstructionsWrapper"] li p::text').getall()

        yield {
            'title': title,
            'ingredients': ingredients,
            'instructions': instructions
        }
