import asyncio
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Thay đổi dòng này bằng token của bạn
TOKEN = "6702157131:AAHLopyJhuNij-Qkez5YySW7td__L31zzHM"

# TOKEN = "aaa"
# Thay đổi dòng này bằng ID chat của bạn
CHAT_ID = "5607828483"

# async def start(update: Update, context: CallbackContext) -> None:
#     user = update.effective_user
#     await update.message.reply_markdown_v2(
#         fr'Hi {user.mention_markdown_v2()}\!',
#         reply_markup=None,
#     )

async def check_token_and_chat():
    try:
        # Kiểm tra token và ID chat
        bot = Bot(token=TOKEN)
        chat = await bot.get_chat(chat_id=CHAT_ID)

        print(f"Token and chat ID are valid and active.")
        print(f"Bot name: {await bot.get_me()}")
        print(f"Chat title: {chat.title}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Token or chat ID might be invalid or the bot is not a member of the channel.")

if __name__ == "__main__":
    # Tạo một event loop mới
    loop = asyncio.get_event_loop()

    # Gọi hàm kiểm tra token và chat ID trong event loop
    loop.run_until_complete(check_token_and_chat())


