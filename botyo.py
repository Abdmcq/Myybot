import logging
import os
import tempfile
import re
import requests
import json
import asyncio
from datetime import datetime

from telegram import Update, ReplyKeyboardRemove, Poll, User
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    PicklePersistence,
)
from PyPDF2 import PdfReader
from http.server import BaseHTTPRequestHandler, HTTPServer # For simple health check endpoint

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Bot Configuration (Embedded for personal use as requested) ---
TELEGRAM_BOT_TOKEN = "7892395794:AAEUNB1UygFFcCbl7vxoEvH_DFGhjkfOlg8" # استبدل بالتوكن الخاص بك
GEMINI_API_KEY = "AIzaSyCtGuhftV0VQCWZpYS3KTMWHoLg__qpO3g" # استبدل بمفتاحك
OWNER_ID = 1749717270  # <--- !!! استبدل هذا بمعرف مستخدم تيليجرام الرقمي الخاص بك !!!
OWNER_USERNAME = "ll7ddd" # استبدل باسم المستخدم الخاص بك في تيليجرام
BOT_PROGRAMMER_NAME = "عبدالرحمن حسن"
PERSISTENCE_FILE = "bot_data_persistence.pkl"

# Conversation states
ASK_NUM_QUESTIONS_FOR_EXTRACTION = range(1)
MCQS_FILENAME = "latest_mcqs.json"

# Webhook configuration for Render (these still need to be environment variables or derived)
# Render provides a PORT environment variable automatically
PORT = int(os.environ.get("PORT", 8000))
WEBHOOK_PATH = "/telegram-webhook" # A specific path for your webhook
# WEBHOOK_URL will be provided by Render or set manually in environment variables
# It's crucial for Render to know its own public URL
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def generate_mcqs_text_blob_with_gemini(text_content: str, num_questions: int, language: str = "Arabic") -> str:
    """
    Generates multiple-choice questions (MCQs) from text content using the Gemini API.
    The output is a single text blob formatted for parsing.
    """
    api_model = "gemini-1.5-flash-latest"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent?key={GEMINI_API_KEY}"
    max_chars = 20000
    text_content = text_content[:max_chars] if len(text_content) > max_chars else text_content

    prompt = f"""
    Generate exactly {num_questions} MCQs in {language} from the text below.
    The questions should aim to comprehensively cover the key information and concepts from the entire provided text.

    STRICT FORMAT (EACH PART ON A NEW LINE):
    Question: [Question text, can be multi-line ending with ? or not]
    A) [Option A text]
    B) [Option B text]
    C) [Option C text]
    D) [Option D text]
    Correct Answer: [Correct option letter, e.g., A, B, C, or D]
    --- (Separator, USED BETWEEN EACH MCQ, BUT NOT after the last MCQ)

    Text:
    \"\"\"
    {text_content}
    \"\"\"
    CRITICAL INSTRUCTIONS:
    1. Each question MUST have exactly 4 options (A, B, C, D). Do not generate questions with fewer than 4 options.
    2. Ensure question text is 10-290 characters long.
    3. Ensure each option text (A, B, C, D) is 1-90 characters long.
    4. The "Correct Answer:" line is CRITICAL and must be present for every MCQ.
    5. The "Correct Answer:" must be one of A, B, C, or D, corresponding to one of the provided options.
    6. Distractor options (incorrect answers) should be plausible but clearly incorrect based on the text.
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        generated_text_candidate = response.json().get("candidates")
        if generated_text_candidate and len(generated_text_candidate) > 0:
            content_parts = generated_text_candidate[0].get("content", {}).get("parts")
            if content_parts and len(content_parts) > 0:
                generated_text = content_parts[0].get("text", "")
                logger.debug(f"Gemini RAW response (first 500 chars): {generated_text[:500]}")
                return generated_text.strip()
        logger.error(f"Gemini API response missing expected structure. Response: {response.json()}")
        return ""
    except requests.exceptions.Timeout:
        logger.error(f"Gemini API request timed out after 300 seconds for {num_questions} questions.")
        return ""
    except Exception as e:
        logger.error(f"Gemini API error: {e}", exc_info=True)
        if hasattr(e, 'response') and e.response is not None: logger.error(f"Gemini Response: {e.response.text}")
        return ""

mcq_parsing_pattern = re.compile(
    r"Question:\s*(.*?)\s*\n"
    r"A\)\s*(.*?)\s*\n"
    r"B\)\s*(.*?)\s*\n"
    r"C\)\s*(.*?)\s*\n"
    r"D\)\s*(.*?)\s*\n"
    r"Correct Answer:\s*([A-D])",
    re.IGNORECASE | re.DOTALL
)

async def send_single_mcq_as_poll(mcq_text: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Sends a single MCQ as a Telegram quiz poll."""
    match = mcq_parsing_pattern.fullmatch(mcq_text.strip())
    if not match:
        logger.warning(f"Could not parse MCQ block for poll (format mismatch or not 4 options):\n-----\n{mcq_text}\n-----")
        return False
    try:
        question_text = match.group(1).strip()
        option_a_text = match.group(2).strip()
        option_b_text = match.group(3).strip()
        option_c_text = match.group(4).strip()
        option_d_text = match.group(5).strip()
        correct_answer_letter = match.group(6).upper()

        options = [option_a_text, option_b_text, option_c_text, option_d_text]

        if not (1 <= len(question_text) <= 300):
            logger.warning(f"Poll Question text too long/short ({len(question_text)} chars): \"{question_text[:50]}...\"")
            return False
        valid_options_for_poll = True
        for i, opt_text in enumerate(options):
            if not (1 <= len(opt_text) <= 100):
                logger.warning(f"Poll Option {i+1} text too long/short ({len(opt_text)} chars): \"{opt_text[:50]}...\" for question \"{question_text[:50]}...\"")
                valid_options_for_poll = False
                break
        if not valid_options_for_poll: return False

        correct_option_id = -1
        letter_to_id = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        if correct_answer_letter in letter_to_id:
            correct_option_id = letter_to_id[correct_answer_letter]

        if correct_option_id == -1:
            logger.error(f"Invalid correct_answer_letter '{correct_answer_letter}'. MCQ:\n{mcq_text}")
            return False

        await context.bot.send_poll(
            chat_id=update.effective_chat.id,
            question=question_text,
            options=options,
            type=Poll.QUIZ,
            correct_option_id=correct_option_id,
            is_anonymous=True,
        )
        return True
    except Exception as e:
        logger.error(f"Error creating poll from MCQ block: {e}\nMCQ:\n{mcq_text}", exc_info=True)
        return False

# --- Owner Restriction and Notification Logic ---
async def handle_restricted_access(update: Update, context: ContextTypes.DEFAULT_TYPE, attempted_feature_name: str = "ميزة محظورة"):
    """
    Logs the access attempt, notifies the owner if it's a new user,
    and sends a restricted access message to the user.
    """
    user = update.effective_user
    if not user:
        return

    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    if 'attempted_users' not in context.bot_data:
        context.bot_data['attempted_users'] = {}

    is_new_attempting_user = False
    if user.id not in context.bot_data['attempted_users']:
        context.bot_data['attempted_users'][user.id] = {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name or "N/A",
            "username": user.username or "N/A",
            "first_attempt_timestamp": now_str
        }
        is_new_attempting_user = True

    if is_new_attempting_user:
        attempt_count = len(context.bot_data['attempted_users'])
        message_to_owner = (
            f"⚠️ محاولة وصول جديدة للبوت (محاولة استخدام: {attempted_feature_name}):\n\n"
            f"👤 المستخدم رقم: {attempt_count}\n"
            f"الاسم الأول: {user.first_name}\n"
            f"الاسم الثاني: {user.last_name or 'لا يوجد'}\n"
            f"المعرف: @{user.username or 'لا يوجد'}\n"
            f"الأيدي: `{user.id}`\n"
            f"تاريخ الدخول: {now_str}"
        )
        try:
            await context.bot.send_message(chat_id=OWNER_ID, text=message_to_owner, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Failed to send owner notification for user {user.id}: {e}")

    await update.message.reply_text(
        f"عذراً، هذا البوت يعمل بشكل حصري لمبرمجه {BOT_PROGRAMMER_NAME} (@{OWNER_USERNAME}).\n"
        "لا يمكنك استخدام وظائفه حالياً."
    )

# --- Command and Conversation Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    user = update.effective_user
    if user.id != OWNER_ID:
        await handle_restricted_access(update, context, attempted_feature_name="/start command")
        return

    await update.message.reply_html(
        rf"مرحباً {update.effective_user.mention_html()}!\n"
        rf"أرسل ملف PDF لاستخراج أسئلة منه. الأسئلة ستُحول إلى اختبارات (quiz polls) مع 4 خيارات لكل سؤال، وتُحفظ كنص في ملف."
    )

async def handle_pdf_for_extraction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles PDF document uploads for MCQ extraction."""
    user = update.effective_user
    if user.id != OWNER_ID:
        await handle_restricted_access(update, context, attempted_feature_name="PDF Upload")
        return ConversationHandler.END

    document = update.message.document
    if not document.mime_type == "application/pdf":
        await update.message.reply_text("من فضلك أرسل ملف PDF صالح.")
        return ConversationHandler.END

    await update.message.reply_text("تم استلام ملف PDF. جاري معالجة النص...")
    try:
        pdf_file = await document.get_file()
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, document.file_name or "temp.pdf")
            await pdf_file.download_to_drive(custom_path=pdf_path)
            text_content = extract_text_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error handling document: {e}", exc_info=True)
        await update.message.reply_text("حدث خطأ أثناء معالجة الملف.")
        return ConversationHandler.END

    if not text_content.strip():
        await update.message.reply_text("لم أتمكن من استخراج أي نص من ملف PDF.")
        return ConversationHandler.END

    context.user_data['pdf_text_for_extraction'] = text_content
    await update.message.reply_text("النص استخرج. كم سؤال (quiz poll) بأربعة خيارات تريد؟ مثال: 5. يمكنك طلب أي عدد (مثلاً 50).")
    return ASK_NUM_QUESTIONS_FOR_EXTRACTION

async def num_questions_for_extraction_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes the number of questions requested by the user."""
    try:
        num_questions_str = update.message.text
        if not num_questions_str.isdigit():
            await update.message.reply_text("الرجاء إرسال رقم صحيح موجب لعدد الأسئلة.")
            return ASK_NUM_QUESTIONS_FOR_EXTRACTION

        num_questions = int(num_questions_str)

        if num_questions < 1:
            await update.message.reply_text("الرجاء إدخال رقم موجب (1 أو أكثر) لعدد الأسئلة.")
            return ASK_NUM_QUESTIONS_FOR_EXTRACTION

        if num_questions > 50:
            await update.message.reply_text(
                f"لقد طلبت إنشاء {num_questions} اختباراً (4 خيارات لكل سؤال). "
                "قد تستغرق هذه العملية بعض الوقت. سأبذل قصارى جهدي!"
            )
        elif num_questions > 20:
             await update.message.reply_text(
                f"جاري تجهيز {num_questions} اختباراً (4 خيارات لكل سؤال). قد يستغرق هذا بضع لحظات..."
            )

    except ValueError:
        await update.message.reply_text("الرجاء إرسال رقم صحيح موجب لعدد الأسئلة.")
        return ASK_NUM_QUESTIONS_FOR_EXTRACTION

    pdf_text = context.user_data.pop('pdf_text_for_extraction', None)
    if not pdf_text:
        await update.message.reply_text("خطأ: نص PDF غير موجود. أعد إرسال الملف.")
        return ConversationHandler.END

    await update.message.reply_text(f"جاري استخراج {num_questions} سؤالاً (4 خيارات لكل سؤال) وتحويلها إلى اختبارات...", reply_markup=ReplyKeyboardRemove())

    generated_mcq_text_blob = generate_mcqs_text_blob_with_gemini(pdf_text, num_questions)

    if not generated_mcq_text_blob:
        await update.message.reply_text("لم أتمكن من استخراج أسئلة من النموذج باستخدام Gemini API.")
        return ConversationHandler.END

    individual_mcqs_texts = [
        mcq.strip() for mcq in re.split(r'\s*---\s*', generated_mcq_text_blob)
        if mcq.strip() and "Correct Answer:" in mcq and "D)" in mcq
    ]

    if not individual_mcqs_texts:
        await update.message.reply_text("لم يتمكن Gemini من إنشاء أسئلة بالتنسيق المطلوب (4 خيارات) أو النص المستخرج فارغ.")
        logger.warning(f"Gemini blob did not yield valid 4-option MCQs: {generated_mcq_text_blob[:300]}")
        return ConversationHandler.END

    actual_generated_count = len(individual_mcqs_texts)
    if actual_generated_count < num_questions:
        await update.message.reply_text(
            f"تم طلب {num_questions} اختباراً، ولكن تمكنت من إنشاء {actual_generated_count} اختباراً فقط بالتنسيق المطلوب (4 خيارات). "
            "قد يكون هذا بسبب طبيعة النص المدخل أو استجابة Gemini."
        )

    try:
        with open(MCQS_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(individual_mcqs_texts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {actual_generated_count} MCQs text (4-options) to {MCQS_FILENAME}")
        await update.message.reply_text(f"تم حفظ نصوص {actual_generated_count} سؤال في `{MCQS_FILENAME}`.\n"
                                        "جاري الآن إنشاء اختبارات (quiz polls)...", parse_mode='Markdown')
    except IOError as e:
        logger.error(f"Could not write to {MCQS_FILENAME}: {e}")
        await update.message.reply_text(f"فشل حفظ نصوص الأسئلة في ملف. سأحاول إنشاء {actual_generated_count} اختباراً.")

    polls_created_count = 0
    delay_between_polls = 0.25

    for mcq_text_item in individual_mcqs_texts:
        if await send_single_mcq_as_poll(mcq_text_item, update, context):
            polls_created_count += 1

        if actual_generated_count > 10:
            await asyncio.sleep(delay_between_polls)

    final_message = f"انتهت العملية.\n"
    final_message += f"تم إنشاء {polls_created_count} اختبار (quiz poll) بنجاح (من أصل {actual_generated_count} سؤال تم إنشاؤه بواسطة Gemini بالتنسيق المطلوب)."
    if polls_created_count < actual_generated_count:
        final_message += f"\nتعذر إنشاء {actual_generated_count - polls_created_count} اختبار بسبب مشاكل في التنسيق لم يتم التعرف عليها أو حدود تيليجرام."

    await update.message.reply_text(final_message)

    return ConversationHandler.END

async def cancel_extraction_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the current MCQ extraction conversation."""
    user = update.effective_user
    if user.id != OWNER_ID:
        await handle_restricted_access(update, context, attempted_feature_name="/cancel command")
        return ConversationHandler.END

    await update.message.reply_text("تم إلغاء العملية.", reply_markup=ReplyKeyboardRemove())
    context.user_data.clear()
    return ConversationHandler.END

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays statistics about bot usage and restricted access attempts."""
    user = update.effective_user
    if user.id != OWNER_ID:
        await handle_restricted_access(update, context, attempted_feature_name="/stats command")
        return

    attempted_users_dict = context.bot_data.get('attempted_users', {})

    if not attempted_users_dict:
        await update.message.reply_text("لم يحاول أي مستخدم الدخول إلى البوت حتى الآن.")
        return

    total_users = len(attempted_users_dict)

    base_message = f"📊 إحصائيات محاولات الوصول للبوت:\n\n"
    base_message += f"إجمالي عدد المستخدمين الذين حاولوا الدخول: {total_users}\n\n"
    base_message += "قائمة المستخدمين:\n"

    messages_to_send = [base_message]
    current_message_part = ""

    for i, (uid, user_data) in enumerate(attempted_users_dict.items()):
        user_entry = (
            f"{i + 1}. الاسم: {user_data.get('first_name', 'N/A')} {user_data.get('last_name', 'N/A')}\n"
            f"   المعرف: @{user_data.get('username', 'N/A')}\n"
            f"   الأيدي: `{uid}`\n"
            f"   أول محاولة: {user_data.get('first_attempt_timestamp', 'N/A')}\n"
            f"--------------------\n"
        )
        if len(current_message_part) + len(user_entry) > 4000:
            messages_to_send.append(current_message_part)
            current_message_part = user_entry
        else:
            current_message_part += user_entry

    if current_message_part:
        messages_to_send.append(current_message_part)

    first_message_sent = False
    for msg_part in messages_to_send:
        if msg_part.strip():
            if not first_message_sent and msg_part == base_message and len(messages_to_send) > 1 and messages_to_send[1].strip():
                await update.message.reply_text(msg_part, parse_mode='Markdown')
                first_message_sent = True
            elif first_message_sent or (msg_part == base_message and len(messages_to_send) == 1):
                await update.message.reply_text(msg_part, parse_mode='Markdown')
            elif not first_message_sent and msg_part != base_message:
                await update.message.reply_text(msg_part, parse_mode='Markdown')
                first_message_sent = True

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Logs errors and sends a user-friendly message."""
    logger.error(f"Update {update} caused error {context.error}", exc_info=True)
    if update and update.effective_message:
        if isinstance(context.error, TelegramError) and "message to edit not found" in str(context.error).lower(): return
        try:
            if update.effective_user and update.effective_user.id == OWNER_ID:
                 await update.effective_message.reply_text(f"عذراً، حدث خطأ ما: {context.error}")
            else:
                 await update.effective_message.reply_text("عذراً، حدث خطأ ما داخلياً.")
        except Exception as e_reply:
            logger.error(f"Error sending error message: {e_reply}")

# --- Health Check Endpoint ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Bot is running!")
        else:
            self.send_response(404)
            self.end_headers()

def run_health_check_server():
    """Starts a simple HTTP server for health checks."""
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    logger.info(f"Starting health check server on port {PORT}")
    httpd.serve_forever()

async def main() -> None:
    """Main function to run the bot."""
    # Create persistence object
    persistence = PicklePersistence(filepath=PERSISTENCE_FILE)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).persistence(persistence).build()

    # Set webhook
    if WEBHOOK_URL:
        await application.bot.set_webhook(url=f"{WEBHOOK_URL}{WEBHOOK_PATH}")
        logger.info(f"Webhook set to: {WEBHOOK_URL}{WEBHOOK_PATH}")
    else:
        logger.warning("WEBHOOK_URL environment variable not set. Webhook might not be configured correctly. "
                       "Please set WEBHOOK_URL in Render environment variables after first deploy.")

    extraction_conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Document.PDF, handle_pdf_for_extraction)],
        states={
            ASK_NUM_QUESTIONS_FOR_EXTRACTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, num_questions_for_extraction_received)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_extraction_command)],
        conversation_timeout=1200
    )
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(extraction_conv_handler)

    application.add_error_handler(error_handler)

    logger.info(f"Bot started. Owner ID: {OWNER_ID}. MCQs will be saved to {MCQS_FILENAME}. Persistence file: {PERSISTENCE_FILE}.")

    # Start the webhook server
    webserver = application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=WEBHOOK_PATH,
        webhook_url=f"{WEBHOOK_URL}{WEBHOOK_PATH}" if WEBHOOK_URL else None # Use WEBHOOK_URL if available
    )

    # Run a separate thread for the simple health check server
    import threading
    health_check_thread = threading.Thread(target=run_health_check_server)
    health_check_thread.daemon = True
    health_check_thread.start()

    await webserver

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)


