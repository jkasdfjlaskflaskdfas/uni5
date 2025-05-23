# main_bot.py (robust, linear Q&A, send_message only)
import logging
import os
import json
import re  # Make sure re is imported at the top
from pathlib import Path
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, constants
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from card_formatters import format_recommend_card, format_details_card
import joblib
from telegram.error import TelegramError

# --- Logger Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- State Constants ---
(
    SELECT_ACTION, ASK_NAME, ASK_QUESTION, CONFIRM_INPUTS, SHOW_RESULTS, SHOW_DETAILS
) = range(6)

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDERS = [
    BASE_DIR / "BTB",
    BASE_DIR / "PP",
    BASE_DIR / "SR",
]
MODEL_PATH = BASE_DIR / 'major_recommender_model.pkl'
ENCODERS_PATH = BASE_DIR / 'label_encoders.pkl'
TELEGRAM_BOT_TOKEN = "8198268528:AAHuAvp3bRqO5hdPD3tbv1xSks2YjXtiu9Y"
WELCOME_STICKER = None  # Disable sticker for robustness
CURRENT_LANG = 'kh'

# --- Language Strings (Khmer) ---
LANG_STRINGS = {
    'kh': {
        'welcome_message': "👋 *សួស្តី!* ខ្ញុំជា *MajorFinder Bot* 🎓\nសូមស្វាគមន៍មកកាន់ដំណើរស្វែងរកជំនាញសិក្សាដ៏ល្អបំផុតសម្រាប់អ្នក! 🚀\n\nតើអ្នកត្រៀមខ្លួនហើយឬនៅ?",
        'start_button': "🎉 ចាប់ផ្តើមឥឡូវនេះ!",
        'about_button': "ℹ️ អំពីបុត",
        'about_text': (
            "*🎓 អំពី MajorFinder Bot*\n\n"
            "ខ្ញុំជាជំនួយការ AI ដែលបង្កើតឡើងដើម្បីជួយសិស្សកម្ពុជាស្វែងរកជំនាញសាកលវិទ្យាល័យដែលស័ក្តិសមបំផុត។ "
            "ជាមួយទិន្នន័យពីសាកលវិទ្យាល័យ និងជំនាញជាច្រើន ខ្ញុំនឹងណែនាំអ្នកដោយផ្អែកលើចំណាប់អារម្មណ៍ និងគោលដៅរបស់អ្នក។\n\n"
            "តោះស្វែងរកអនាគតរបស់អ្នកជាមួយគ្នា! 😊"
        ),
        'ask_name': "😊 តោះស្គាល់គ្នាបន្តិច! តើខ្ញុំអាចហៅអ្នកថាអ្វី?",
        'name_received': "👋 សួស្តី *{name}*! តោះចាប់ផ្តើមជាមួយសំណួរដំបូង (1/8) 🌟",
        'ask_interests': "🎨 តើអ្នកចូលចិត្តធ្វើអ្វីក្នុងពេលទំនេរ? (1/8)",
        'int_sports': "🏀 កីឡា", 'int_tech': "💻 បច្ចេកវិទ្យា", 'int_arts': "🎨 សិល្បៈ",
        'int_social': "🤝 សកម្មភាពសង្គម", 'int_other': "🌟 ផ្សេងៗ (សូមបញ្ជាក់)",
        'ask_subjects': "📚 មុខវិជ្ជាណាដែលអ្នកចូលចិត្ត ឬធ្វើបានល្អជាងគេ? (2/8)",
        'sub_math': "➕ គណិត", 'sub_science': "🔬 វិទ្យាសាស្ត្រ", 'sub_lang': "📖 ភាសា",
        'sub_social': "🌍 សង្គម", 'sub_tech': "🖥️ បច្ចេកវិទ្យា",
        'ask_problem_style': "🤓 ស្រមៃថាអ្នកត្រូវដោះស្រាយបញ្ហាធំមួយ តើអ្នកនឹងធ្វើយ៉ាងណា? (3/8)",
        'prob_analyze': "📊 វិភាគទិន្ននន័យ", 'prob_create': "🎨 ច្នៃប្រឌិតថ្មី",
        'prob_team': "👥 សុំជំនួយក្រុម", 'prob_trial': "🛠️ សាកល្បងផ្ទាល់",
        'ask_work_env': "🏢 តើអ្នកចូលចិត្តបរិយាកាសការងារបែបណា? (4/8)",
        'env_team': "👥 ក្រុមរស់រវើក", 'env_quiet': "🧘 ស្ងប់ស្ងាត់ឯករាជ្យ",
        'env_flex': "⚖️ លាយឡំទាំងពីរ",
        'ask_career_goal': "🚀 តើអ្នកសុបិនចង់ក្លាយជាអ្វីនាពេលអនាគត? (5/8)",
        'goal_expert': "🏆 អ្នកជំនាញ", 'goal_leader': "🌟 អ្នកដឹកនាំ",
        'goal_innovator': "✨ អ្នកបង្កើតថ្មី", 'goal_helper': "💖 អ្នកជួយសង្គម",
        'ask_location': "🌍 តើអ្នកចង់សិក្សានៅទីណា? (6/8)",
        'loc_phnompenh': "🏙️ ភ្នំពេញ", 'loc_province': "🏞️ ខេត្ត", 'loc_any': "🌏 ទីណាក៏បាន",
        'ask_budget': "💰 តើថវិកាសម្រាប់ថ្លៃសិក្សារបស់អ្នកប្រហែលប៉ុន្មានក្នុងមួយឆ្នាំ? (7/8)",
        'bud_low': "💸 < $1000", 'bud_mid': "💵 $1000 - $3000",
        'bud_high': "💳 > $3000", 'bud_flexible': "🤔 មិនទាន់សម្រេចចិត្ត",
        'ask_learning': "📖 តើអ្នកចូលចិត្តរៀនបែបណា? (8/8)",
        'learn_practice': "🛠️ អនុវត្តផ្ទាល់", 'learn_read': "📚 អាន/ស្តាប់",
        'learn_group': "👥 ក្រុម", 'learn_online': "💻 អនឡាញ",
        'confirm_title': "📝 *សូមបញ្ជាក់ចម្លើយរបស់អ្នក*",
        'confirm_text': "នេះជាចម្លើយរបស់អ្នក៖\n\n{summary}\n\nតើត្រឹមត្រូវទេ? បើត្រឹមត្រូវ ខ្ញុំនឹងផ្តល់ជំនាញណែនាំភ្លាម! 😊",
        'confirm_yes': "✅ ត្រឹមត្រូវ បន្តទៅមុខ!",
        'confirm_no': "🔙 កែចម្លើយ",
        'processing': "⏳ *កំពុងវិភាគ...*\nសូមរង់ចាំបន្តិច ខ្ញុំកំពុងស្វែងរកជំនាញដ៏ល្អបំផុតសម្រាប់អ្នក!",
        'results_title': "✨ *លទ្ធផលណែនាំសម្រាប់ {name}* ✨",
        'results_intro': "ខាងក្រោមនេះជាជំនាញដែលស័ក្តិសមនឹងអ្នកបំផុត ដោយផ្អែកលើចម្លើយរបស់អ្នក៖",
        'more_details': "ℹ️ ព័ត៌មានបន្ថែម",
        'start_over': "🔄 ចាប់ផ្តើមឡើងវិញ",
        'end': "🏁 បញ្ចប់",
        'back': "⬅️ ត្រឡប់",
        'error': "⚠️ អូ! មានបញ្ហាបន្តិច។ សូមចាប់ផ្តើមឡើងវិញជាមួយ /start ឬទាក់ទងអ្នកបង្កើត។",
        'no_data': "😔 សូមអភ័យទោស រកមិនឃើញព័ត៌មានលម្អិតសម្រាប់ជំនាញនេះទេ។ តើអ្នកចង់ចាប់ផ្តើមឡើងវិញទេ?",
        'no_recommendation': "😔 សូមអភ័យទោស ដោយផ្អែកលើចម្លើយរបស់អ្នក ខ្ញុំមិនអាចរកជំនាញដែលត្រូវគ្នាល្អឥតខ្ចោះបានទេ។ តើអ្នកចង់ព្យាយាមម្តងទៀតដោយកែសម្រួលចម្លើយខ្លះទេ?",
        'thank_you': "🙏 *សូមអរគុណ {name}!* សូមជូនពរឱ្យអ្នកជោគជ័យក្នុងការសិក្សា! 🎉",
    }
}
def _(key, **kwargs):
    return LANG_STRINGS[CURRENT_LANG].get(key, f"<Missing lang: {key}>").format(**kwargs) if kwargs else LANG_STRINGS[CURRENT_LANG].get(key, f"<Missing lang: {key}>")

def escape_markdown_v2(text: str) -> str:
    """Correctly escapes special characters for MarkdownV2 using re.sub."""
    if not isinstance(text, str):
        text = str(text)
    # Characters that must be escaped for MarkdownV2
    raw_chars_to_escape = r'_*[]()~`>#+-=|{}.!'
    pattern = f"([{re.escape(raw_chars_to_escape)}])"
    return re.sub(pattern, r'\\\1', text)

# --- Data Loading ---
def load_all_majors(data_folders_paths):
    all_majors_list = []
    for folder_path in data_folders_paths:
        if not folder_path.exists():
            logger.warning(f"Data folder not found: {folder_path}")
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                try:
                    with open(folder_path / filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_majors_list.append(data)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
    logger.info(f"Loaded a total of {len(all_majors_list)} major entries from all folders.")
    return all_majors_list

majors_data = []
model = None
label_encoders = None

def load_app_data():
    global majors_data, model, label_encoders
    majors_data = load_all_majors(DATA_FOLDERS)
    try:
        if MODEL_PATH.exists() and ENCODERS_PATH.exists():
            model = joblib.load(MODEL_PATH)
            label_encoders = joblib.load(ENCODERS_PATH)
            logger.info("ML Model and encoders loaded successfully.")
        else:
            logger.error(f"Model ({MODEL_PATH}) or Encoders ({ENCODERS_PATH}) file not found. Please run train_model.py first.")
    except Exception as e:
        logger.error(f"Error loading ML model/encoders: {e}")

# --- Q&A Flow Definition ---
QUESTIONS = [
    {
        'key': 'hobby',
        'text': 'ask_interests',
        'options': [
            ('int_sports', 'Sports'),
            ('int_tech', 'Technology'),
            ('int_arts', 'Arts & Culture'),
            ('int_social', 'Social Activities'),
            ('int_other', 'Others'),
        ]
    },
    {
        'key': 'subject_enjoyment',
        'text': 'ask_subjects',
        'options': [
            ('sub_math', 'Mathematics'),
            ('sub_science', 'Natural Sciences'),
            ('sub_lang', 'Languages'),
            ('sub_social', 'Social Sciences'),
            ('sub_tech', 'IT/Engineering'),
        ]
    },
    {
        'key': 'problem_approach',
        'text': 'ask_problem_style',
        'options': [
            ('prob_analyze', 'Analytical'),
            ('prob_create', 'Creative'),
            ('prob_team', 'Collaborative'),
            ('prob_trial', 'Hands-on Trial'),
        ]
    },
    {
        'key': 'work_environment',
        'text': 'ask_work_env',
        'options': [
            ('env_team', 'Team-oriented'),
            ('env_quiet', 'Quiet/Independent'),
            ('env_flex', 'Flexible/Mixed'),
        ]
    },
    {
        'key': 'longterm_goal',
        'text': 'ask_career_goal',
        'options': [
            ('goal_expert', 'Become a Specialist'),
            ('goal_leader', 'Lead a Team/Company'),
            ('goal_innovator', 'Innovate/Create New Things'),
            ('goal_helper', 'Help Others/Society'),
        ]
    },
    {
        'key': 'decision_making',
        'text': 'ask_location',
        'options': [
            ('loc_phnompenh', 'Phnom Penh'),
            ('loc_province', 'Provinces'),
            ('loc_any', 'Anywhere'),
        ]
    },
    {
        'key': 'team_role',
        'text': 'ask_budget',
        'options': [
            ('bud_low', '<$1000/year'),
            ('bud_mid', '$1000-$3000/year'),
            ('bud_high', '>$3000/year'),
            ('bud_flexible', 'Flexible/Undecided'),
        ]
    },
    {
        'key': 'learning_style',
        'text': 'ask_learning',
        'options': [
            ('learn_practice', 'Practical/Hands-on'),
            ('learn_read', 'Reading/Lectures'),
            ('learn_group', 'Group Discussions'),
            ('learn_online', 'Online Courses'),
        ]
    },
]

# --- Keyboard Helper ---
def create_keyboard(options, question_idx=None):
    keyboard = []
    for i in range(0, len(options), 2):
        row = []
        for btn_key, btn_val in options[i:i+2]:
            callback_data = f"q{question_idx}:{btn_val}" if question_idx is not None else btn_val
            row.append(InlineKeyboardButton(_(btn_key), callback_data=callback_data))
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)

# --- Conversation Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data['answers'] = {}
    context.user_data['q_idx'] = 0
    await update.message.reply_text(
        escape_markdown_v2(_('welcome_message')),
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton(_('start_button'), callback_data='action_start_survey')],
            [InlineKeyboardButton(_('about_button'), callback_data='action_show_about')],
        ]),
        parse_mode=constants.ParseMode.MARKDOWN_V2
    )
    return SELECT_ACTION

async def select_action_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    logger.info(f"select_action_handler: callback_data={getattr(query, 'data', None)}")
    await query.answer()
    if query.data == 'action_show_about':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=escape_markdown_v2(_('about_text')),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(_('start_button'), callback_data='action_start_survey')],
            ]),
            parse_mode=constants.ParseMode.MARKDOWN_V2
        )
        return SELECT_ACTION
    elif query.data == 'action_start_survey':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=escape_markdown_v2(_('ask_name')),
            parse_mode=constants.ParseMode.MARKDOWN_V2
        )
        return ASK_NAME
    else:
        logger.error(f"Unknown callback data in select_action_handler: {query.data}")
        await context.bot.send_message(chat_id=query.message.chat_id, text=_('error'))
        return SELECT_ACTION

async def select_action_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info(f"select_action_message_handler: text={update.message.text}")
    await update.message.reply_text(
        escape_markdown_v2(_('welcome_message')),
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton(_('start_button'), callback_data='action_start_survey')],
            [InlineKeyboardButton(_('about_button'), callback_data='action_show_about')],
        ]),
        parse_mode=constants.ParseMode.MARKDOWN_V2
    )
    return SELECT_ACTION

async def name_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    name = update.message.text.strip()
    context.user_data['user_name'] = name
    context.user_data['q_idx'] = 0
    await update.message.reply_text(
        escape_markdown_v2(_('name_received', name=name)),
        parse_mode=constants.ParseMode.MARKDOWN_V2
    )
    return await ask_question(update, context)

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q_idx = context.user_data.get('q_idx', 0)
    if q_idx >= len(QUESTIONS):
        return await confirm_inputs(update, context)
    q = QUESTIONS[q_idx]
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=escape_markdown_v2(_(q['text'])),
        reply_markup=create_keyboard(q['options'], question_idx=q_idx),
        parse_mode=constants.ParseMode.MARKDOWN_V2
    )
    return ASK_QUESTION

async def answer_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith('q'):
        try:
            q_idx, answer_val = data[1:].split(':', 1)
            q_idx = int(q_idx)
        except Exception:
            await context.bot.send_message(chat_id=query.message.chat_id, text=_('error'))
            return ConversationHandler.END
        if q_idx != context.user_data.get('q_idx', 0):
            # Ignore out-of-order presses
            return ASK_QUESTION
        q = QUESTIONS[q_idx]
        context.user_data['answers'][q['key']] = answer_val
        context.user_data['q_idx'] = q_idx + 1
        if context.user_data['q_idx'] >= len(QUESTIONS):
            return await confirm_inputs(update, context)
        else:
            return await ask_question(update, context)
    else:
        await context.bot.send_message(chat_id=query.message.chat_id, text=_('error'))
        return ConversationHandler.END

async def confirm_inputs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    answers = context.user_data.get('answers', {})
    summary = []
    for i, q in enumerate(QUESTIONS):
        val = answers.get(q['key'], '-')
        # Try to get display value from lang strings
        display = _(val) if val in LANG_STRINGS[CURRENT_LANG] else val
        summary.append(f"*{escape_markdown_v2(_(q['text']))}*: {escape_markdown_v2(display)}")
    summary_text = '\n'.join(summary)
    text = escape_markdown_v2(_('confirm_title')) + "\n\n" + escape_markdown_v2(_('confirm_text', summary=summary_text))
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton(_('confirm_yes'), callback_data='confirm_yes')],
            [InlineKeyboardButton(_('confirm_no'), callback_data='confirm_no')],
        ]),
        parse_mode=constants.ParseMode.MARKDOWN_V2
    )
    return CONFIRM_INPUTS

async def process_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == 'confirm_no':
        context.user_data['q_idx'] = 0
        return await ask_question(update, context)
    elif query.data == 'confirm_yes':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=escape_markdown_v2(_('processing')),
            parse_mode=constants.ParseMode.MARKDOWN_V2
        )
        # --- AI Prediction Logic ---
        if not model or not label_encoders:
            await context.bot.send_message(chat_id=query.message.chat_id, text=_('error'))
            return ConversationHandler.END
        user_answers = context.user_data.get('answers', {})
        # Prepare input for the model (must match your model's expected columns)
        try:
            input_df = prepare_model_input(user_answers)
            preds = model.predict(input_df)
            # For demo, just show the first result
            major = preds[0] if len(preds) > 0 else None
            if not major:
                await context.bot.send_message(chat_id=query.message.chat_id, text=_('no_recommendation'))
                return ConversationHandler.END
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=escape_markdown_v2(_('results_title', name=context.user_data.get('user_name', '')))+"\n"+escape_markdown_v2(_('results_intro')),
                parse_mode=constants.ParseMode.MARKDOWN_V2
            )
            # Format and send the recommendation card
            card = format_recommend_card(major)
            await context.bot.send_message(chat_id=query.message.chat_id, text=card, parse_mode=constants.ParseMode.MARKDOWN_V2)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=_('thank_you', name=context.user_data.get('user_name', '')),
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton(_('start_over'), callback_data='action_start_survey')],
                ]),
                parse_mode=constants.ParseMode.MARKDOWN_V2
            )
            return SHOW_RESULTS
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            await context.bot.send_message(chat_id=query.message.chat_id, text=_('error'))
            return ConversationHandler.END
    else:
        await context.bot.send_message(chat_id=query.message.chat_id, text=_('error'))
        return ConversationHandler.END

def prepare_model_input(user_answers):
    # This function must match your model's expected input columns and encodings
    import pandas as pd
    # Example: columns = [q['key'] for q in QUESTIONS]
    columns = [q['key'] for q in QUESTIONS]
    row = [user_answers.get(col, '') for col in columns]
    df = pd.DataFrame([row], columns=columns)
    # Apply label encoders if needed
    for col in columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    return df

# --- Global Error Handler ---
async def error_handler(update, context):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if update and hasattr(update, 'effective_chat') and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Internal error. Please try again later."
            )
        except Exception:
            pass

# --- Catch-all CallbackQueryHandler for unmatched callback queries ---
async def unknown_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    logger.warning(f"Unknown callback query received: {getattr(query, 'data', None)}")
    await query.answer()
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="⚠️ Unknown action. Please use /start to begin again."
    )
    return ConversationHandler.END

# --- Main ---
def main():
    load_app_data()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(error_handler)
    logger.info("Bot is starting up...")
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start_command)],
        states={
            SELECT_ACTION: [
                CallbackQueryHandler(select_action_handler, pattern=r'^action_'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, select_action_message_handler),
                CallbackQueryHandler(unknown_callback_handler),
            ],
            ASK_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, name_handler)],
            ASK_QUESTION: [CallbackQueryHandler(answer_handler)],
            CONFIRM_INPUTS: [CallbackQueryHandler(process_confirmation)],
            SHOW_RESULTS: [CallbackQueryHandler(select_action_handler)],
        },
        fallbacks=[CommandHandler('start', start_command)],
        allow_reentry=True,  # Allow users to restart
    )
    app.add_handler(conv_handler)
    logger.info("Bot started. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == '__main__':
    main()