const nutritionix = require("nutritionix-api");
const fs = require('fs')
// import fetch from "node-fetch"
const YOUR_API_KEY = '05e560d00a23a5155bf96db7f7805b9d'
const YOUR_APP_ID = '6a6ba3f4'
nutritionix.init(YOUR_APP_ID, YOUR_API_KEY);



const classFood = ["hot-dog", "Apple", "Artichoke", "Asparagus", "Bagel", "pie", "Banana", "Beer", "Bell-pepper", "Bread",
   "Broccoli", "Burrito", "Cabbage", "Cake", "Candy", "Cantaloupe", "Carrot", "Fresh-fig", "Cookie", "cendol-dessert",
   "French-fries", "Grape", "Guacamole", "Hot-dog", "Ice-cream", "Muffin", "Orange", "Pancake", "Pear", "Popcorn",
   "Pretzel", "Strawberry", "Tomato", "Waffle", "soft-drinks", "Cheese", "Cocktail", "Coffee",
   "Cooking-spray", "Crab", "Croissant", "Cucumber", "Doughnut", "Egg", "fresh-fruit", "Grapefruit", "Hamburger", "Honeycomb",
   "Juice", "Lemon", "Lobster", "Mango", "Milk", "Mushroom", "Oyster", "Pasta", "Pastry", "Peach",
   "Pineapple", "Pizza", "Pomegranate", "Potato", "Pumpkin", "Radish", "Salad", "indian-food", "Sandwich", "Shrimp",
   "Squash", "Squid", "turkey-sandwich", "Sushi", "Taco", "Tart", "Tea", "Vegetable", "Watermelon", "Wine",
   "Winter-melon", "Zucchini"]

// hot-dog', 'apple', 'Artichoke', 'Bagel' ,'Cookie', 'Dessert', 'French-fries', 'Grape', 'Guacamole', 'Hot-dog', 'Ice-cream', 'Muffin', 'Orange', 'Pancake',
//    'Pear', 'Popcorn', 'Pretzel', 'Strawberry', 'Tomato', 'Waffle', 'food-drinks', 'Cheese', 'Cocktail', 'Coffee', 'Cooking-spray', 'Crab', 'Croissant', 'Cucumber',
//    'Doughnut', 'Egg', 'Fruit', 'Grapefruit', 'Hamburger', 'Honeycomb', 'Juice', 'Lemon', 'Lobster', 'Mango', 'Milk', 'Mushroom', 'Oyster', 'Pasta', 'Pastry', 'Peach',
//    'Pineapple', 'Pizza', 'Pomegranate', 'Potato', 'Pumpkin', 'Radish', 'Salad', 'food', 'Sandwich', 'Shrimp', 'Squash', 'Squid', 'Submarine-sandwich', 'Sushi', 'Taco',
//    'Tart', 'Tea', 'Vegetable', 'Watermelon', 'Wine', 'Winter-melon', 'Zucchini', 'Banh_mi', 'Banh_trang_tron', 'Banh_xeo', 'Bun_bo_Hue',
//    'Bun_dau', 'Com_tam', 'Goi_cuon', 'Pho', 'Hu_tieu', 'Xoi'

//Dessert,Baked-goods,Common-fig,"food-drinks",food
const getFactFood = async (classFood) => {
   const results = []
   for (const index in classFood) {
      let result = await nutritionix.natural.search(classFood[index])
      result = result.foods[0]
      nf = ['nf_total_fat', 'nf_total_carbohydrate', 'nf_dietary_fiber', 'nf_protein', 'nf_sugars']
      const final = {
         "food_name": result.food_name,
         "nf": [...nf],
         "value": nf.map(item => result[item]),
         "serving_weight_grams": result.serving_weight_grams,
         "nf_calories": result.nf_calories,
         "id": index,
         // "nf_total_fat": result.nf_total_fat,
         // "nf_total_carbohydrate": result.nf_total_carbohydrate,
         // "nf_sodium": result.nf_sugars,
         // "nf_protein": result.nf_protein,
      }
      results.push(final)
   }
   return results
}


getFactFood(classFood)
   .then(data => {

      fs.writeFile("./food.json", JSON.stringify(data, null, 4), (err) => {
         if (err) {
            console.error(err);
            return;
         };
         console.log("File has been created");
      });
      // console.log(data[0]);
   })
   .catch(err => { console.log(err); })



// Honeycomb, Asparagus, Crab, Waffle, Lobster, Pasta, Popcorn, Milk, Squash, Mushroom, Cake, Burrito, Winter-melon,
// Bell-pepper, Ice-cream, Baked-goods, Pear, Hot-dog, Pumpkin, Cocktail, Coffee, Taco, Pizza, Oyster, Grape, Pineapple,
// Candy, Guacamole, Sandwich, Egg, Broccoli, Submarine-sandwich, Squid, Sushi, Salad, Banana, Watermelon, Tart, Strawberry,
// Juice, Bread, Peach, Apple, Pancake, Artichoke, Cabbage, Beer, Common-fig, Croissant, Hamburger, Cheese, Dessert, Fruit, Radish,
// Pastry, Cantaloupe, Tomato, Pomegranate, Shrimp, Wine, Cooking-spray, Potato, Lemon, Doughnut, Vegetable, French-fries, Orange,
// Carrot, Cucumber, Mango, Grapefruit, Tea, Zucchini, Bagel