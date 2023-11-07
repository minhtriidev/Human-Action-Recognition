import asyncio
from telegram import Bot
from telegram import InputFile

bot = Bot(token="6702157131:AAHLopyJhuNij-Qkez5YySW7td__L31zzHM")
chat_id = "5607828483"

image_path = "D:\\Documents\\Picture_D\\Ảnh linh tinh\\wallpaperflare.com_wallpaper (3).jpg"
message = "This is your alert message with an image from nitro5"

async def send_telegram_message_with_image():
    # Upload the image and get the file ID
    with open(image_path, "rb") as img:
        image = InputFile(img)
        await bot.send_photo(chat_id=chat_id, photo=image, caption=message)

# Tạo một event loop và chạy coroutine với thời gian chờ
loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(asyncio.wait_for(send_telegram_message_with_image(), timeout=60))
except asyncio.TimeoutError:
    print("Timeout: The operation took too long to complete.")


# import asyncio
# from telegram import Bot

# bot = Bot(token="YOUR_BOT_TOKEN")
# chat_id = "YOUR_CHAT_ID"

# message = "This is your alert message"

# async def send_telegram_message():
#     await bot.send_message(chat_id=chat_id, text=message)

# # Tạo một event loop và chạy coroutine
# loop = asyncio.get_event_loop()
# loop.run_until_complete(send_telegram_message())
