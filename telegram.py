import logging
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.filters import Command
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import asyncio
import requests
import psycopg2
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

API_TOKEN = '7065747281:AAHFsGOtVRGA-xlhLFsChdbTHlw4ND4uIwE'
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'
GEMINI_API_KEY = 'AIzaSyBfnTWkR_iDTWguXcTa8k9nuf5WiWeY7aM'

# Database connection details
DB_USERNAME = "kitaphub_owner"
DB_PASSWORD = "oxFAO2lU8CJg"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize bot, dispatcher, and storage
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())

data = [
    {'user_id': -1, 'book_id': 1271159, 'rating': 5, 'title': '1984'},
    {'user_id': -1, 'book_id': 23492624, 'rating': 3, 'title': 'brother'},
    {'user_id': -1, 'book_id': 834713, 'rating': 5, 'title': 'the thousand splendid suns'},
    {'user_id': -1, 'book_id': 27157078, 'rating': 1, 'title': 'cristiano'},
    {'user_id': -1, 'book_id': 819495, 'rating': 5, 'title': 'the kite runner'},
    {'user_id': -1, 'book_id': 11297, 'rating': 5, 'title': 'norwegian wood'},
    {'user_id': -1, 'book_id': 5107, 'rating': 5, 'title': 'The Catcher in the Rye'}
]

# Create a DataFrame from the array
my_books = pd.DataFrame(data)
my_books.set_index('user_id', inplace=True)


# Define states for the FSM
class Form(StatesGroup):
    choosing_function = State()
    ai_recommendation = State()
    kitaphub_login = State()
    kitaphub_recommendation = State()


# Define start command handler
@dp.message_handler(Command("start"))
async def start_command(message: types.Message):
    await message.answer(
        "Добро пожаловать в наш чат, который создан для KitapHub. Выберите функции, которые хотите использовать:\n"
        "1. ai_recommend - рекомендации в чате с искусственным интеллектом\n"
        "2. kitaphub_recommendation - рекомендации в чате с подбором")
    await Form.choosing_function.set()


# Define stop command handler
@dp.message_handler(Command("stop"), state="*")
async def stop_command(message: types.Message, state: FSMContext):
    await message.answer("Спасибо за использование!")
    await state.finish()


# Define handler for choosing function
@dp.message_handler(state=Form.choosing_function)
async def choose_function(message: types.Message, state: FSMContext):
    choice = message.text.lower()
    if choice == "ai_recommend":
        await message.answer("Введите ваш любимый жанр:")
        await Form.ai_recommendation.set()
    elif choice == "kitaphub_recommendation":
        await message.answer("Пожалуйста, введите ваш логин для KitapHub:")
        await Form.kitaphub_login.set()
    else:
        await message.answer("Неправильный выбор. Попробуйте еще раз.")
        await Form.choosing_function.set()


# Define handler for AI recommendations
@dp.message_handler(state=Form.ai_recommendation)
async def ai_recommend(message: types.Message, state: FSMContext):
    favorite_genre = message.text
    response = call_gemini_api(favorite_genre)
    await message.answer(response)

    await message.answer("Выберите следующие действия:\n"
                         "1. ai_recommend - рекомендации в чате с искусственным интеллектом\n"
                         "2. kitaphub_recommendation - рекомендации в чате с подбором\n"
                         "Или завершите чат с помощью команды /stop")
    await Form.choosing_function.set()


def call_gemini_api(genre):
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "contents": [{"parts": [{"text": f"Назовите пять книг в жанре {genre}, на русском языке"}]}]
        }
        response = requests.post(f'{GEMINI_API_URL}?key={GEMINI_API_KEY}', headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Ошибка: Кандидаты не найдены в ответе"
    except requests.HTTPError as http_error:
        return f"HTTP ошибка: {http_error}"
    except Exception as e:
        return f"Ошибка вызова API Gemini: {e}"


# Define handler for KitapHub login
@dp.message_handler(state=Form.kitaphub_login)
async def kitap_hub_login(message: types.Message, state: FSMContext):
    username = message.text
    if validate_login(username):
        await message.answer("login success")

        # Retrieve book IDs for the user
        book_ids = get_book_ids(username)
        if book_ids:
            data = [{'user_id': -1, 'book_id': book_id, 'rating': 5, 'title': 'idk'} for book_id in book_ids]
            my_books = pd.DataFrame(data)
            my_books.set_index('user_id', inplace=True)
            file_book_id = r'ml_data\book_id_map.csv'
            file_goodread_int = r'ml_data\filtered_goodreads_interactions.csv'
            file_book_titles = r'ml_data\books_titles.json'

            interactions = load_data(file_book_id, file_goodread_int)
            ratings_mat, interactions = process_data(interactions)

            my_index = interactions[interactions["user_id"] == "-1"]["user_index"].unique()[0]
            top_recs = recommend_books(ratings_mat, interactions, my_index, file_book_titles)
            translated_titles = translate_with_gemini(top_recs)
            await message.answer("Here are your book recommendation from our webpage:")
            await message.answer(translated_titles)
        else:
            await message.answer("No books found for your account.")

        await message.answer("Выберите следующие действия:\n"
                             "1. ai_recommend - рекомендации в чате с искусственным интеллектом\n"
                             "2. kitaphub_recommendation - рекомендации в чате с подбором\n"
                             "Или завершите чат с помощью команды /stop")
        await Form.choosing_function.set()
    else:
        await message.answer("username неправильный, попробуйте еще раз")
        await Form.kitaphub_login.set()


def validate_login(username):
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname="kitaphub",
            user=DB_USERNAME,
            password=DB_PASSWORD,
            host="ep-soft-limit-a2t2akkp.eu-central-1.aws.neon.tech",
            sslmode="require"
        )
        cursor = conn.cursor()
        query = "SELECT username FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except Exception as e:
        logger.error(f"Database error: {e}")
        return False


def get_book_ids(username):
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname="kitaphub",
            user=DB_USERNAME,
            password=DB_PASSWORD,
            host="ep-soft-limit-a2t2akkp.eu-central-1.aws.neon.tech",
            sslmode="require"
        )
        cursor = conn.cursor()
        query = """
            SELECT b.good_reads_book_id 
            FROM books b
            JOIN user_books ub ON b.isbn = ub.isbn
            JOIN users u ON ub.user_id = u.id
            WHERE u.username = %s
        """
        cursor.execute(query, (username,))
        book_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return book_ids
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []


def load_data(file_book_id, file_goodread_int):
    my_books["book_id"] = my_books["book_id"].astype(str)

    csv_book_mapping = {}
    with open(file_book_id, "r") as f:
        for line in f:
            csv_id, book_id = line.strip().split(",")
            csv_book_mapping[csv_id] = book_id

    book_set = set(my_books["book_id"])

    overlap_users = {}
    with open(file_goodread_int) as f:
        for line in f:
            user_id, csv_id, _, rating, _ = line.strip().split(",")
            book_id = csv_book_mapping.get(csv_id)
            if book_id in book_set:
                overlap_users[user_id] = overlap_users.get(user_id, 0) + 1

    filtered_overlap_users = set(k for k, v in overlap_users.items() if v > my_books.shape[0] / 5)

    interactions_list = []
    with open(file_goodread_int) as f:
        for line in f:
            user_id, csv_id, _, rating, _ = line.strip().split(",")
            if user_id in filtered_overlap_users:
                book_id = csv_book_mapping[csv_id]
                interactions_list.append([user_id, book_id, rating])

    interactions = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])
    interactions = pd.concat([my_books.reset_index()[["user_id", "book_id", "rating"]], interactions])

    interactions["book_id"] = interactions["book_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"])

    return interactions


def process_data(interactions):
    interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
    interactions["book_index"] = interactions["book_id"].astype("category").cat.codes

    ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))
    return ratings_mat_coo.tocsr(), interactions


def recommend_books(ratings_mat, interactions, my_index, file_book_titles):
    similarity = cosine_similarity(ratings_mat[my_index, :], ratings_mat).flatten()
    indices = np.argpartition(similarity, -600)[-600:]

    similar_users = interactions[interactions["user_index"].isin(indices)].copy()
    similar_users = similar_users[similar_users["user_id"] != "-1"]

    book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])

    books_titles = pd.read_json(file_book_titles)
    books_titles["book_id"] = books_titles["book_id"].astype(str)
    book_recs = book_recs.merge(books_titles, how="inner", on="book_id")

    book_recs["adjusted_count"] = book_recs["count"] * (book_recs["count"] / book_recs["mean"])
    book_recs["score"] = book_recs["mean"] * book_recs["adjusted_count"]

    book_recs = book_recs[~book_recs["book_id"].isin(my_books["book_id"])]

    my_books["mod_title"] = my_books["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True).str.lower()
    book_recs = book_recs[~book_recs["book_id"].isin(my_books["mod_title"])]
    book_recs = book_recs[book_recs["count"] > 2]
    book_recs = book_recs[book_recs["mean"] > 1]

    top_10_titles = book_recs["title"].head(10).tolist()
    return top_10_titles


def translate_with_gemini(text):
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "contents": [{"parts": [{"text": f"Сделайте перевод для этих книг {text}, на русском языке"}]}]
        }
        response = requests.post(f'{GEMINI_API_URL}?key={GEMINI_API_KEY}', headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Ошибка: Кандидаты не найдены в ответе"
    except requests.HTTPError as http_error:
        return f"HTTP ошибка: {http_error}"
    except Exception as e:
        return f"Ошибка вызова API Gemini: {e}"


async def main():
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
