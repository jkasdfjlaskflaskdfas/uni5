import json
import logging
import joblib
import re
import datetime
import asyncio
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# --- Logger Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def escape_markdown_v2(text: str) -> str:
    """Escapes special characters for MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- State Constants ---
(
    SELECT_ACTION, ASK_NAME,
    ASK_CURRENT_STATUS, # New
    ASK_SUBJECT_STRENGTH, # Replaces ASK_ACADEMIC_STRENGTHS, uses v2 questions
    ASK_PROBLEM_SOLVING_STYLE, ASK_WORK_ENVIRONMENT_PREFERENCE, ASK_IMPACT_ASPIRATION, # From v1
    ASK_STUDY_LOCATION, # Replaces ASK_LOCATION_PREFERENCE, uses v2 questions
    ASK_BUDGET_RANGE, # From v1
    ASK_BAC_RESULT, # New
    ASK_LEARN_STYLE, # New
    ASK_EXTRA_ACTIVITY, # New
    ASK_SCHOLARSHIP, # New
    CONFIRM_INPUTS_AND_PROCESS, SHOW_RESULTS, SHOW_DETAILS, AWAITING_FEEDBACK
) = range(17) # CORRECTED: Adjusted range for new states


# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDERS = [
    Path("/Users/admin/Downloads/Database/BTB"), # Please ensure these paths are correct
    Path("/Users/admin/Downloads/Database/PP"),
    Path("/Users/admin/Downloads/Database/SR"),
]
MODEL_PATH = BASE_DIR / 'major_recommender_model.pkl'
ENCODERS_PATH = BASE_DIR / 'label_encoders.pkl'

TELEGRAM_BOT_TOKEN = "8198268528:AAHuAvp3bRqO5hdPD3tbv1xSks2YjXtiu9Y"
BOT_USERNAME = "teslajds1bot"

THANK_YOU_STICKER_ID = "CAACAgIAAxkBAAEsjpZmdA3oF_ym0WzcjAnn3aQ4pEvsywACUQADwZxgDGWWbQYb4u9qMAQ"

# --- Language Strings (Khmer) - Updated with new questions ---
LANG_STRINGS = {
    'kh': {
        'welcome_message': "👋 សួស្តី! ខ្ញុំជាជំនួយការ AI របស់អ្នកក្នុងការស្វែងរកជំនាញសាកលវិទ្យាល័យ។\n\nតើអ្នកចង់ចាប់ផ្តើមដំណើរស្វែងរកជំនាញដែលសក្តិសមនឹងអ្នកទេ?",
        'start_assessment_button': "🚀 ចាប់ផ្តើមស្វែងរក!",
        'about_bot_button_label': "ℹ️ អំពីខ្ញុំ",
        'main_menu_button': "🏠 មឺនុយគោល",
        'back_button': "⬅️ ត្រឡប់",
        'cancel_message': "🚫 ការសន្ទនាត្រូវបានបោះបង់។ \nវាយ /start ដើម្បីចាប់ផ្តើមម្តងទៀតគ្រប់ពេល!",
        'thank_you_message': "🙏 សូមអរគុណដែលបានប្រើប្រាស់សេវាកម្មរបស់ខ្ញុំ! សូមជូនពរអ្នកឱ្យទទួលបានជោគជ័យក្នុងការសិក្សា!",
        'error_message': "⚠️ មានបញ្ហាបន្តិចបន្តួច។ សូមព្យាយามម្តងទៀត ឬទាក់ទងអ្នកអភិវឌ្ឍន៍។",
        'error_loading_model': "⚠️ មានបញ្ហាក្នុងការផ្ទុកប្រព័ន្ធណែនាំ។ មុខងារណែនាំអាចមានកំណត់។",

        'about_bot_title': "ℹ️ អំពីជំនួយការ AI នេះ",
        'about_bot_text': (
            "ខ្ញុំត្រូវបានបង្កើតឡើងដើម្បីជួយណែនាំអ្នកក្នុងការជ្រើសរើសជំនាញសិក្សានៅសាកលវិទ្យាល័យ។ "
            "តាមរយៈសំណួរមួយចំនួន ខ្ញុំនឹងព្យាយាមស្វែងយល់ពីចំណាប់អារម្មណ៍ និងគោលដៅរបស់អ្នក រួចផ្តល់ជាយោបល់។\n\n"
            "បច្ចុប្បន្ន ខ្ញុំប្រើប្រាស់ទិន្នន័យពីសាកលវិទ្យាល័យមួយចំនួនក្នុងប្រទេសកម្ពុជា និងប្រព័ន្ធ Machine Learning ដើម្បីជួយអ្នក។"
            "\n\nបេសកកម្មរបស់ខ្ញុំគឺធ្វើឱ្យការសម្រេចចិត្តរបស់អ្នកកាន់តែងាយស្រួល និងមានទំនុកចិត្ត! 😊"
        ),
        'ask_name_prompt': "😊 តោះចាប់ផ្តើម! ដើម្បីឱ្យការសន្ទនារបស់យើងកាន់តែស្និទ្ធស្នាល តើខ្ញុំអាចហៅអ្នកឈ្មោះអ្វីបានទេ?",
        'name_received_prompt': "សួស្តី {name}! 👋 ខ្ញុំត្រៀមខ្លួនរួចរាល់ហើយក្នុងការជួយអ្នកស្វែងរកជំនាញដ៏ល្អបំផុត។",

        'ask_current_status': "ស្ថានភាពបច្ចុប្បន្នរបស់អ្នក?",
        'status_grade12': "សិស្សថ្នាក់ទី ១២",
        'status_graduated_highschool': "បញ្ចប់វិទ្យាល័យ",
        'status_university_student': "កំពុងសិក្សានៅសាកលវិទ្យាល័យ",
        'status_changing_major': "ផ្លាស់ប្តូរជំនាញសិក្សា",
        'status_other': "ផ្សេងៗ",

        'section_academic_title': "📚 តោះស្វែងយល់ពីផ្នែកសិក្សារបស់អ្នក...",
        'ask_subject_strength': "សូមជ្រើសរើសមុខវិជ្ជាដែលអ្នកមានភាពងាយស្រួលរៀនបំផុត?",
        'subject_math': "គណិតវិទ្យា",
        'subject_physics': "រូបវិទ្យា",
        'subject_chemistry': "គីមីវិទ្យា",
        'subject_biology': "ជីវវិទ្យា",
        'subject_khmer_lang': "ភាសាខ្មែរ",
        'subject_english_lang': "ភាសាអង់គ្លេស",
        'subject_ict_computer': "ICT/កុំព្យូទ័រ",
        'subject_other_subjects': "មុខវិជ្ជាផ្សេងៗ",

        'section_approach_title': "🤔 តោះពិចារណាពីរបៀបដែលអ្នកចូលចិត្តដោះស្រាយបញ្ហា និងរៀនសូត្រ...",
        'ask_problem_solving_style': "នៅពេលប្រឈមមុខនឹងបញ្ហា តើអ្នកចូលចិត្តដោះស្រាយវាដោយរបៀបណា? (ជ្រើសរើសមួយដែលសមស្របបំផុត)",
        'style_analytical': "វិភាគទិន្នន័យ និងស្វែងរកដំណោះស្រាយឡូជីខល 📊",
        'style_creative': "គិតគូរពីគំនិតថ្មីៗ និងច្នៃប្រឌិត 🎨",
        'style_practical': "អនុវត្តជាក់ស្តែង និងបង្កើតអ្វីមួយ 🛠️",
        'style_collaborative': "សហការជាមួយអ្នកដទៃដើម្បីរកដំណោះស្រាយរួម 🤝",

        'ask_work_environment_preference': "តើអ្នកចូលចិត្តបរិយាកាសការសិក្សា ឬការងារនាពេលអនាគតបែបណា? (ជ្រើសរើសមួយ)",
        'env_dynamic_team': "បរិយាកាសក្រុមដែលរស់រវើក និងពោរពេញដោយគំនិត 👥",
        'env_focused_independent': "កន្លែងស្ងប់ស្ងាត់សម្រាប់ធ្វើការងារដោយឯករាជ្យ និងផ្ចង់អារម្មណ៍ 🧘",
        'env_mixed_collaboration_solo': "ការผสมผสานរវាងការសហការ និងការងារតែម្នាក់ឯង ⚖️", # User to translate "ผสมผสาน"

        'section_aspirations_title': "🚀 តោះនិយាយពីគោលដៅ និងក្តីស្រមៃរបស់អ្នក...",
        'ask_impact_aspiration': "តើអ្នកចង់មានឥទ្ធិពលបែបណាដែរតាមរយៈអាជីពរបស់អ្នក? (ជ្រើសរើសមួយ)",
        'impact_expert': "ក្លាយជាអ្នកជំនាញឈានមុខគេក្នុងสาขาเฉพาะทาง 🏆", # User to translate "ในสาขาเฉพาะทาง"
        'impact_leader': "ដឹកនាំ និងជម្រុញក្រុមឱ្យบรรลุเป้าหมายធំៗ 🌟", # User to translate "บรรลุเป้าหมายធំៗ"
        'impact_innovator': "បង្កើតนวัตกรรม និងนำเสนอគំនិត/ផលិតផលថ្មីៗ ✨", # User to translate "นวัตกรรม"
        'impact_social_contributor': "រួមចំណែកដោះស្រាយបញ្ហាសង្គមសំខាន់ៗ 💖",

        'section_practicalities_title': "🌍💰 ជាចុងក្រោយ តោះពិចារណាពីปัจจัยមួយចំនួន...", # User to translate "ปัจจัยមួយចំនួន"
        'ask_study_location': "តើអ្នកចង់សិក្សានៅទីក្រុង ឬខេត្ត?",
        'location_phnom_penh': "ទីក្រុង (ភ្នំពេញ)",
        'location_suburban': "តំបន់ជាយក្រុង",
        'location_province_rural': "ខេត្ត/ជនបទ",
        'location_no_preference': "គ្មានចំណូលចិត្ត",

        'ask_budget_range': "ចំពោះថ្លៃសិក្សា តើកម្រិតថវិកាប្រចាំปีរបស់អ្នកស្ថិតនៅក្នុងកម្រិតណា? (ជ្រើសរើសមួយ)",
        'budget_affordable': "សមរម្យ (ឧ. < $1000/ปี) �",
        'budget_mid_range': "មធ្យម (ឧ. $1000 - $3000/ปี) 💵",
        'budget_flexible_high': "អាចបត់បែនបាន/ខ្ពស់ (ឧ. > $3000/ปี) 💳",

        'ask_bac_result': "លទ្ធផលសញ្ញាបត្រមធ្យមសិក្សាទុតិយភូមិ (Bac II) របស់អ្នក?",
        'bac_a': "កម្រិត A", 'bac_b': "កម្រិត B", 'bac_c': "កម្រិត C",
        'bac_d': "កម្រិត D", 'bac_e': "កម្រិត E", 'bac_not_yet': "មិនទាន់ប្រឡង",

        'ask_learn_style': "តើអ្នកចូលចិត្តរៀនតាមរបៀបណាជាងគេ?",
        'learn_reading_listening': "អានសៀវភៅ/ស្តាប់ការบรรยาย", # User to translate "บรรยาย"
        'learn_practical': "រៀនតាមការអនុវត្ត",
        'learn_group': "រៀនជាក្រុម",
        'learn_online': "រៀនតាមវីដេអូ/អ៊ីនធឺណិត",
        'learn_workshop_project': "រៀនតាមសិក្ខាសាលា/គម្រោង",
        'learn_style_other': "ផ្សេងៗ",

        'ask_extra_activity': "តើអ្នកចូលចិត្តសកម្មភាពក្រៅម៉ោងអ្វី?",
        'activity_sports': "កីឡា", 'activity_music_art': "តន្ត្រី/សិល្បៈ",
        'activity_coding_tech': "Coding/បច្ចេកវិទ្យា", 'activity_volunteer': "សកម្មភាពស្ម័គ្រចិត្ត",
        'activity_small_business': "ធ្វើអាជីវកម្មតូចៗ", 'activity_none': "មិនសូវចូលចិត្ត",
        'activity_other': "ផ្សេងៗ",

        'ask_scholarship': "តើអ្នកត្រូវការអាហារូបករណ៍ដែរឬទេ?",
        'scholarship_needed': "ត្រូវការ", 'scholarship_not_needed': "មិនចាំបាច់",
        'scholarship_not_sure': "មិនប្រាកដ",

        'confirm_inputs_title': "📝 សូមตรวจสอบព័ត៌មានរបស់អ្នក៖",
        'confirm_inputs_text': "នេះជាចម្លើយដែលអ្នកបានផ្តល់៖\n\n{answers_summary}\n\nតើព័ត៌មានទាំងអស់ត្រឹមត្រូវទេ? ប្រសិនបើត្រឹមត្រូវ យើងនឹងចាប់ផ្តើមវិភាគដើម្បីស្វែងរកការណែនាំដ៏ល្អបំផុតសម្រាប់អ្នក។",
        'confirm_button': "✅ បាទ ត្រឹមត្រូវ! ចាប់ផ្តើមវិភាគ",

        'processing_answers': "⏳ កំពុងដំណើរការចម្លើយរបស់អ្នក... សូមរង់ចាំបន្តិច។ ខ្ញុំកំពុងពិគ្រោះជាមួយប្រព័ន្ធឆ្លាតវៃរបស់ខ្ញុំ! 😊",
        'results_title': "✨ លទ្ធផលការណែនាំសម្រាប់អ្នក ✨",
        'results_intro': "ដោយផ្អែកលើចម្លើយរបស់អ្នក នេះគឺជាជំនាញមួយចំនួនដែលអាចនឹងសមស្របជាមួយអ្នក៖",
        'results_no_match_model': "😔 អូ! ប្រព័ន្ធណែនាំរបស់ខ្ញុំហាក់ដូចជាមិនទាន់មានព័ត៌មានត្រូវនឹងចំណូលចិត្តរបស់អ្នកទាំងស្រុងទេ។ ប៉ុន្តែខ្ញុំអាចបង្ហាញជំនាញពេញនិយមមួយចំនួន។",
        'results_no_match_data': "😔 សូមអភ័យទោស យើងមិនអាចរកឃើញព័ត៌មានជំនាញដែលត្រូវគ្នានៅក្នុងមូលដ្ឋានទិន្នន័យរបស់យើងទេ។",
        'results_what_next': "តើអ្នកចង់ធ្វើអ្វីបន្តទៀត?",
        'learn_more_button_label': "ℹ️ ស្វែងយល់បន្ថែម",
        'show_another_suggestion_button': "🔄 បង្ហាញការណែនាំផ្សេងទៀត",
        'start_over_button': "🔄 ចាប់ផ្តើមម្តងទៀត",
        'end_conversation_button': "🏁 បញ្ចប់ការសន្ទនា",

        'major_details_title': "📖 ព័ត៌មានលម្អិតអំពីជំនាញ៖ {major_kh}",
        'major_details_university': "🏢 សាកលវិទ្យាល័យ៖ {uni_kh} ({uni_en})",
        'major_details_faculty': "🎓 មហាវិទ្យាល័យ៖ {faculty_kh} ({faculty_en})",
        'major_details_location': "📍 ទីតាំង៖ {location}",
        'major_details_description': "📝 ការពិពណ៌នា៖\n{description}",
        'major_details_keywords': "🔑 ពាក្យគន្លឹះ៖ {keywords}",
        'major_details_career_prospects': "💼 អាជីពនាពេលអនាគត៖\n{careers}",
        'major_details_core_subjects': "📚 មុខវិជ្ជាស្នូល៖\n{core_subjects}",
        'major_details_link': "🌐 តំណភ្ជាប់បន្ថែម៖ {link}",
        'no_details_available': "សូមអភ័យទោស ព័ត៌មានលម្អិតសម្រាប់ជំនាញនេះមិនទាន់មាននៅឡើយទេ។",
        'back_to_results_button': "⬅️ ត្រឡប់ទៅលទ្ធផល",
    }
}
CURRENT_LANG = 'kh'

def _(key, **kwargs):
    raw = LANG_STRINGS.get(CURRENT_LANG, {}).get(key, f"<Missing: {key}>")
    if kwargs:
        try: return raw.format(**kwargs)
        except KeyError: return f"<Format Err: {key}>"
    return raw

majors_data = []
model = None
label_encoders = None

def load_data_and_model():
    global majors_data, model, label_encoders
    temp_majors_data = []
    for folder_path in DATA_FOLDERS:
        if not folder_path.exists() or not folder_path.is_dir():
            logger.warning(f"Data folder not found or not a directory: {folder_path}")
            continue
        for json_file in folder_path.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'major_en' in data and 'university_en' in data :
                        temp_majors_data.append(data)
                    else:
                        logger.warning(f"Skipping {json_file}: missing 'major_en' or 'university_en'.")
            except json.JSONDecodeError: logger.error(f"Error decoding JSON from {json_file}")
            except Exception as e: logger.error(f"Error loading {json_file}: {e}")
    majors_data = temp_majors_data
    if not majors_data: logger.error("CRITICAL: No majors data loaded.")
    else: logger.info(f"Successfully loaded {len(majors_data)} major entries.")
    try:
        if MODEL_PATH.exists(): model = joblib.load(MODEL_PATH); logger.info(f"Successfully loaded model from {MODEL_PATH}")
        else: logger.error(f"Model file not found at {MODEL_PATH}"); model = None
        if ENCODERS_PATH.exists(): label_encoders = joblib.load(ENCODERS_PATH); logger.info(f"Successfully loaded label encoders from {ENCODERS_PATH}")
        else: logger.error(f"Label encoders file not found at {ENCODERS_PATH}"); label_encoders = None
    except Exception as e:
        logger.error(f"Error loading model or encoders: {e}"); model = None; label_encoders = None
load_data_and_model()

def create_keyboard(buttons_layout, add_back_button_data=None):
    keyboard = []
    for row_layout in buttons_layout:
        row_buttons = []
        for item in row_layout:
            if isinstance(item, tuple) and len(item) == 2:
                text_key, callback_data_val = item
            else:
                text_key = item
                callback_data_val = item
            row_buttons.append(InlineKeyboardButton(_(text_key), callback_data=callback_data_val))
        keyboard.append(row_buttons)
    if add_back_button_data:
         keyboard.append([InlineKeyboardButton(_('back_button'), callback_data=add_back_button_data)])
    return InlineKeyboardMarkup(keyboard)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear(); context.user_data['answers'] = {}; context.user_data['model_inputs'] = {}
    reply_markup = create_keyboard([[('start_assessment_button', 'start_assessment')], [('about_bot_button_label', 'show_about')]])
    if update.message: await update.message.reply_text(_('welcome_message'), reply_markup=reply_markup)
    elif update.callback_query: await update.callback_query.answer(); await update.callback_query.edit_message_text(_('welcome_message'), reply_markup=reply_markup)
    return SELECT_ACTION

async def handle_show_about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    reply_markup = create_keyboard([[('start_assessment_button', 'start_assessment')], [('main_menu_button', 'main_menu')]])
    await query.edit_message_text(f"*{_('about_bot_title')}*\n\n{_('about_bot_text')}", reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
    return SELECT_ACTION

async def ask_name_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    await query.edit_message_text(_('ask_name_prompt'))
    return ASK_NAME

async def handle_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_name = update.message.text; context.user_data['user_name'] = user_name
    await update.message.reply_text(_('name_received_prompt', name=user_name))
    reply_markup = create_keyboard([
        [('status_grade12', 'status_grade12'), ('status_graduated_highschool', 'status_graduated_highschool')],
        [('status_university_student', 'status_university_student')],
        [('status_changing_major', 'status_changing_major'), ('status_other', 'status_other')]
    ], add_back_button_data='back_to_ask_name') # This callback needs a handler if user goes back from name input
    await update.message.reply_text(_('ask_current_status'), reply_markup=reply_markup)
    return ASK_CURRENT_STATUS

CALLBACK_TO_MODEL_VALUE_MAP = {
    'subject_strength': {
        'subject_math': 'Math', 'subject_physics': 'Physics', 'subject_chemistry': 'Chemistry',
        'subject_biology': 'Biology', 'subject_khmer_lang': 'Khmer Language',
        'subject_english_lang': 'English Language', 'subject_ict_computer': 'Computer Science',
        'subject_other_subjects': 'Other',
    },
    'problem_solving_style': {
        'style_analytical': 'Analytical', 'style_creative': 'Creative',
        'style_practical': 'Practical', 'style_collaborative': 'Collaborative',
    },
    'work_environment_preference': {
        'env_dynamic_team': 'Group', 'env_focused_independent': 'Independent',
        'env_mixed_collaboration_solo': 'Mix',
    },
    'impact_aspiration': {
        'impact_expert': 'Expert', 'impact_leader': 'Leader',
        'impact_innovator': 'Innovate', 'impact_social_contributor': 'Social',
    },
    'study_location': {
        'location_phnom_penh': 'City', 'location_suburban': 'Suburban', # Ensure model handles 'Suburban' or map it
        'location_province_rural': 'Province', 'location_no_preference': 'Flexible', # Ensure model handles 'Flexible' or map
    },
    'budget_range': {
        'budget_affordable': 'Low', 'budget_mid_range': 'Medium',
        'budget_flexible_high': 'High',
    }
}
def store_answer(context: ContextTypes.DEFAULT_TYPE, question_key: str, display_text_key_or_val: str, model_feature_name: str = None, map_dict: dict = None):
    user_choice_cb = context.user_data['current_callback_data']
    display_text = _(display_text_key_or_val) if display_text_key_or_val in LANG_STRINGS['kh'] else display_text_key_or_val
    context.user_data.setdefault('answers', {})[question_key + '_display'] = display_text

    if model_feature_name and map_dict:
        model_value = map_dict.get(user_choice_cb)
        if model_value:
            context.user_data.setdefault('model_inputs', {})[model_feature_name] = model_value
            logger.info(f"Q: '{question_key}', UserCB: '{user_choice_cb}', ModelFeat: '{model_feature_name}', ModelVal: '{model_value}'.")
        else:
            logger.warning(f"Q: '{question_key}', UserCB: '{user_choice_cb}', No map for ModelFeat: '{model_feature_name}'.")
    elif model_feature_name: # Direct storage if no map_dict, assuming user_choice_cb is the model value
        context.user_data.setdefault('model_inputs', {})[model_feature_name] = user_choice_cb
        logger.info(f"Q: '{question_key}', UserCB: '{user_choice_cb}', Stored directly for ModelFeat: '{model_feature_name}'.")
    else: # Store as supplementary answer
        context.user_data.setdefault('answers', {})[question_key] = user_choice_cb
        logger.info(f"Q: '{question_key}', UserCB: '{user_choice_cb}', Stored as supplementary answer.")

async def handle_current_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'current_status', query.data) # query.data is like 'status_grade12'
    # await query.edit_message_text(_('section_academic_title')) # This removes the previous question text
    reply_markup = create_keyboard([
        [('subject_math', 'subject_math'), ('subject_physics', 'subject_physics')],
        [('subject_chemistry', 'subject_chemistry'), ('subject_biology', 'subject_biology')],
        [('subject_khmer_lang', 'subject_khmer_lang'), ('subject_english_lang', 'subject_english_lang')],
        [('subject_ict_computer', 'subject_ict_computer'), ('subject_other_subjects', 'subject_other_subjects')]
    ], add_back_button_data='back_to_ask_current_status') # Back to the function that asks current_status
    await query.edit_message_text(f"{_('section_academic_title')}\n\n{_('ask_subject_strength')}", reply_markup=reply_markup)
    return ASK_SUBJECT_STRENGTH

async def handle_subject_strength(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'subject_strength', query.data, 'favorite_subject', CALLBACK_TO_MODEL_VALUE_MAP['subject_strength'])
    reply_markup = create_keyboard([
        [('style_analytical', 'style_analytical'), ('style_creative', 'style_creative')],
        [('style_practical', 'style_practical'), ('style_collaborative', 'style_collaborative')]
    ], add_back_button_data='back_to_ask_subject_strength')
    await query.edit_message_text(f"{_('section_approach_title')}\n\n{_('ask_problem_solving_style')}", reply_markup=reply_markup)
    return ASK_PROBLEM_SOLVING_STYLE

async def handle_problem_solving_style(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'problem_solving_style', query.data, 'study_approach', CALLBACK_TO_MODEL_VALUE_MAP['problem_solving_style'])
    reply_markup = create_keyboard([
        [('env_dynamic_team', 'env_dynamic_team')], [('env_focused_independent', 'env_focused_independent')],
        [('env_mixed_collaboration_solo', 'env_mixed_collaboration_solo')]
    ], add_back_button_data='back_to_ask_problem_solving_style')
    await query.edit_message_text(_('ask_work_environment_preference'), reply_markup=reply_markup)
    return ASK_WORK_ENVIRONMENT_PREFERENCE

async def handle_work_environment_preference(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'work_environment_preference', query.data, 'learning_env', CALLBACK_TO_MODEL_VALUE_MAP['work_environment_preference'])
    reply_markup = create_keyboard([
        [('impact_expert', 'impact_expert'), ('impact_leader', 'impact_leader')],
        [('impact_innovator', 'impact_innovator'), ('impact_social_contributor', 'impact_social_contributor')]
    ], add_back_button_data='back_to_ask_work_environment_preference')
    await query.edit_message_text(f"{_('section_aspirations_title')}\n\n{_('ask_impact_aspiration')}", reply_markup=reply_markup)
    return ASK_IMPACT_ASPIRATION

async def handle_impact_aspiration(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'impact_aspiration', query.data, 'future_goals', CALLBACK_TO_MODEL_VALUE_MAP['impact_aspiration'])
    reply_markup = create_keyboard([
        [('location_phnom_penh', 'location_phnom_penh'), ('location_suburban', 'location_suburban')],
        [('location_province_rural', 'location_province_rural'), ('location_no_preference', 'location_no_preference')]
    ], add_back_button_data='back_to_ask_impact_aspiration')
    await query.edit_message_text(f"{_('section_practicalities_title')}\n\n{_('ask_study_location')}", reply_markup=reply_markup)
    return ASK_STUDY_LOCATION

async def handle_study_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'study_location', query.data, 'location', CALLBACK_TO_MODEL_VALUE_MAP['study_location'])
    reply_markup = create_keyboard([
        [('budget_affordable', 'budget_affordable'), ('budget_mid_range', 'budget_mid_range')],
        [('budget_flexible_high', 'budget_flexible_high')]
    ], add_back_button_data='back_to_ask_study_location')
    await query.edit_message_text(_('ask_budget_range'), reply_markup=reply_markup)
    return ASK_BUDGET_RANGE

async def handle_budget_range(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    logger.info(f"Entering handle_budget_range with callback data: {query.data}")
    store_answer(context, 'budget_range', query.data, 'budget', CALLBACK_TO_MODEL_VALUE_MAP['budget_range'])
    reply_markup = create_keyboard([
        [('bac_a', 'bac_a'), ('bac_b', 'bac_b'), ('bac_c', 'bac_c')],
        [('bac_d', 'bac_d'), ('bac_e', 'bac_e'), ('bac_not_yet', 'bac_not_yet')]
    ], add_back_button_data='back_to_ask_budget_range')
    await query.edit_message_text(_('ask_bac_result'), reply_markup=reply_markup)
    return ASK_BAC_RESULT

async def handle_bac_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'bac_result', query.data)
    reply_markup = create_keyboard([
        [('learn_reading_listening', 'learn_reading_listening'), ('learn_practical', 'learn_practical')],
        [('learn_group', 'learn_group'), ('learn_online', 'learn_online')],
        [('learn_workshop_project', 'learn_workshop_project'), ('learn_style_other', 'learn_style_other')]
    ], add_back_button_data='back_to_ask_bac_result')
    await query.edit_message_text(_('ask_learn_style'), reply_markup=reply_markup)
    return ASK_LEARN_STYLE

async def handle_learn_style(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'learn_style', query.data)
    reply_markup = create_keyboard([
        [('activity_sports', 'activity_sports'), ('activity_music_art', 'activity_music_art')],
        [('activity_coding_tech', 'activity_coding_tech'), ('activity_volunteer', 'activity_volunteer')],
        [('activity_small_business', 'activity_small_business')],
        [('activity_none', 'activity_none'), ('activity_other', 'activity_other')]
    ], add_back_button_data='back_to_ask_learn_style')
    await query.edit_message_text(_('ask_extra_activity'), reply_markup=reply_markup)
    return ASK_EXTRA_ACTIVITY

async def handle_extra_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'extra_activity', query.data)
    reply_markup = create_keyboard([
        [('scholarship_needed', 'scholarship_needed')],
        [('scholarship_not_needed', 'scholarship_not_needed')],
        [('scholarship_not_sure', 'scholarship_not_sure')]
    ], add_back_button_data='back_to_ask_extra_activity')
    await query.edit_message_text(_('ask_scholarship'), reply_markup=reply_markup)
    return ASK_SCHOLARSHIP

async def handle_scholarship(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); context.user_data['current_callback_data'] = query.data
    store_answer(context, 'scholarship', query.data)
    summary_parts = []
    display_order_config = [
        ('current_status_display', "ស្ថានភាពបច្ចុប្បន្ន"),
        ('subject_strength_display', "មុខវិជ្ជាខ្លាំង"),
        ('problem_solving_style_display', "របៀបដោះស្រាយបញ្ហា"),
        ('work_environment_preference_display', "បរិយាកាសការងារ/សិក្សា"),
        ('impact_aspiration_display', "គោលដៅអនាគត"),
        ('study_location_display', "ទីតាំងសិក្សា"),
        ('budget_range_display', "កម្រិតថវិកា"),
        ('bac_result_display', "លទ្ធផលបាក់ឌុប"),
        ('learn_style_display', "របៀបរៀន"),
        ('extra_activity_display', "សកម្មភាពក្រៅម៉ោង"),
        ('scholarship_display', "អាហារូបករណ៍")
    ]
    answers_dict = context.user_data.get('answers', {})
    for key, label_text in display_order_config:
        if key in answers_dict:
            # Values are already translated display texts from store_answer
            value_to_display = answers_dict[key]
            summary_parts.append(f"*{escape_markdown_v2(label_text)}*: {escape_markdown_v2(value_to_display)}")

    answers_summary_text = "\n".join(summary_parts) if summary_parts else _("មិនមានចម្លើយត្រូវបានកត់ត្រាទុក។") # Use _ for default message
    logger.info(f"Summary text length: {len(answers_summary_text)}")
    logger.debug(f"Summary text content: {answers_summary_text}")

    reply_markup = create_keyboard([ [('confirm_button', 'process_answers')], ], add_back_button_data='back_to_ask_scholarship')
    try:
        await query.edit_message_text(
            f"*{escape_markdown_v2(_('confirm_inputs_title'))}*\n\n{_('confirm_inputs_text', answers_summary=answers_summary_text)}",
            reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        logger.error(f"Error editing message in handle_scholarship: {e}", exc_info=True)
        # Fallback or re-send if edit fails critically
        await context.bot.send_message(chat_id=update.effective_chat.id, text=_('error_message') + " Please try /start again.")
        return ConversationHandler.END # Or a specific error state
    return CONFIRM_INPUTS_AND_PROCESS

async def process_answers_and_recommend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    processing_message = await query.edit_message_text(_('processing_answers'))
    context.user_data['last_message_id_before_results'] = processing_message.message_id

    if not model or not label_encoders:
        logger.error("Model or label encoders not loaded.")
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_message.message_id,
                                            text=_('error_loading_model') + "\n" + _('results_what_next'),
                                            reply_markup=create_keyboard([[('main_menu_button', 'main_menu')]]))
        return SHOW_RESULTS

    expected_model_features = ['favorite_subject', 'location', 'study_approach', 'learning_env', 'future_goals', 'budget']
    model_inputs_for_prediction = {}
    all_features_present = True
    user_model_inputs = context.user_data.get('model_inputs', {})
    for feature in expected_model_features:
        if feature not in user_model_inputs:
            logger.error(f"CRITICAL: Missing core model feature '{feature}'.")
            all_features_present = False; break
        model_inputs_for_prediction[feature] = user_model_inputs[feature]

    if not all_features_present:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_message.message_id,
                                            text=_('error_message') + " (Missing critical input for prediction)")
        return SELECT_ACTION

    try:
        input_df = pd.DataFrame([model_inputs_for_prediction], columns=expected_model_features)
        encoded_df = input_df.copy()
        for column in expected_model_features:
            if column in label_encoders:
                le = label_encoders[column]
                try: encoded_df[column] = le.transform(input_df[column])
                except ValueError as e:
                    logger.warning(f"Unseen label for {column}: {input_df[column].iloc[0]}. {e}")
                    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_message.message_id, text=_('error_message') + f" (Could not process {escape_markdown_v2(column)})")
                    return SELECT_ACTION
            else:
                logger.error(f"Label encoder not found for REQUIRED column: {column}.")
                await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_message.message_id, text=_('error_message') + " (Internal error with encoders)")
                return SELECT_ACTION
        prediction = model.predict(encoded_df)
        predicted_major_name = prediction[0]
        logger.info(f"Model prediction: {predicted_major_name}")
        context.user_data['predicted_major'] = predicted_major_name
        recommended_majors_details = [m for m in majors_data if m.get('major_en', '').lower() == predicted_major_name.lower()]
        context.user_data['recommended_majors_list'] = recommended_majors_details
        context.user_data['current_recommendation_index'] = 0
        if not recommended_majors_details:
            reply_markup = create_keyboard([ [('start_over_button', 'start_over')], [('end_conversation_button', 'end_conversation')] ])
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_message.message_id, text=f"{_('results_title')}\n\n{_('results_no_match_data')}\n\n{_('results_what_next')}", reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
        else:
            await display_single_recommendation(update, context, is_initial_display=True)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        reply_markup = create_keyboard([[('main_menu_button', 'main_menu')]])
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_message.message_id, text=_('error_message') + " (Recommendation engine error)\n" + _('results_what_next'), reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
    return SHOW_RESULTS

def format_major_display(major_detail):
    text = f"🎓 *{escape_markdown_v2(major_detail.get('major_kh', major_detail.get('major_en', 'N/A')))}*\n"
    text += f"🏢 _{escape_markdown_v2(major_detail.get('university_kh', major_detail.get('university_en', 'N/A')))}_"
    if major_detail.get('faculty_kh') or major_detail.get('faculty_en'):
        text += f"\n   មហាវិទ្យាល័យ: {escape_markdown_v2(major_detail.get('faculty_kh', major_detail.get('faculty_en', '')))}"
    return text

async def display_single_recommendation(update: Update, context: ContextTypes.DEFAULT_TYPE, is_initial_display=False):
    query = update.callback_query
    if query: await query.answer()
    recommended_majors = context.user_data.get('recommended_majors_list', [])
    current_index = context.user_data.get('current_recommendation_index', 0)
    target_chat_id = update.effective_chat.id
    target_message_id = context.user_data.get('last_message_id_before_results') if is_initial_display else (query.message.message_id if query else None)

    if not recommended_majors:
        no_match_text = f"{_('results_title')}\n\n{_('results_no_match_model')}\n\n{_('results_what_next')}"
        reply_markup = create_keyboard([[('start_over_button', 'start_over')], [('end_conversation_button', 'end_conversation')]])
        if target_message_id:
            try: await context.bot.edit_message_text(chat_id=target_chat_id, message_id=target_message_id, text=no_match_text, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
            except: await context.bot.send_message(chat_id=target_chat_id, text=no_match_text, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
        else: await context.bot.send_message(chat_id=target_chat_id, text=no_match_text, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
        return

    major_to_display = recommended_majors[current_index]
    context.user_data['current_displayed_major_id'] = major_to_display.get('major_en')
    
    display_text_content = f"{_('results_intro')}\n\n{format_major_display(major_to_display)}\n\n{_('results_what_next')}"
    # Title is already bolded by _() if it contains markdown, or we can make it bold here
    final_display_text = f"*{escape_markdown_v2(_('results_title'))}*\n\n{display_text_content}"


    action_buttons_row = [('learn_more_button_label', f"learn_more_{major_to_display.get('major_en', '').replace(' ', '_')}_{major_to_display.get('university_en', '').replace(' ', '_')}")]
    navigation_buttons_row = [('show_another_suggestion_button', 'next_suggestion')] if len(recommended_majors) > 1 else []
    navigation_buttons_row.extend([('start_over_button', 'start_over'), ('end_conversation_button', 'end_conversation')])
    buttons_layout = [action_buttons_row, navigation_buttons_row]
    reply_markup = create_keyboard(buttons_layout)

    if target_message_id:
        try: await context.bot.edit_message_text(chat_id=target_chat_id, message_id=target_message_id, text=final_display_text, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
        except Exception as e: logger.error(f"Error editing results: {e}"); await context.bot.send_message(chat_id=target_chat_id, text=final_display_text, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)
    else: await context.bot.send_message(chat_id=target_chat_id, text=final_display_text, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2)

async def handle_next_suggestion(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    recommended_majors = context.user_data.get('recommended_majors_list', [])
    current_index = context.user_data.get('current_recommendation_index', 0)
    current_index = (current_index + 1) % len(recommended_majors) if recommended_majors else 0
    context.user_data['current_recommendation_index'] = current_index
    await display_single_recommendation(update, context)
    return SHOW_RESULTS

async def handle_learn_more(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer()
    try:
        major_id_to_find = context.user_data.get('current_displayed_major_id')
        if not major_id_to_find: await query.edit_message_text(_('error_message') + " (No major selected)"); return SHOW_RESULTS
        found_major = next((m for m in context.user_data.get('recommended_majors_list', []) if m.get('major_en') == major_id_to_find), None)
        if not found_major: found_major = next((m for m in majors_data if m.get('major_en') == major_id_to_find), None)
        if not found_major: await query.edit_message_text(_('no_details_available')); return SHOW_DETAILS

        details_parts = [f"*{escape_markdown_v2(_('major_details_title', major_kh=found_major.get('major_kh', 'N/A')))}*"]
        details_parts.append(f"{escape_markdown_v2(_('major_details_university', uni_kh=found_major.get('university_kh', 'N/A'), uni_en=found_major.get('university_en', 'N/A')))}")
        if found_major.get('faculty_kh') or found_major.get('faculty_en'): details_parts.append(f"{escape_markdown_v2(_('major_details_faculty', faculty_kh=found_major.get('faculty_kh', ''), faculty_en=found_major.get('faculty_en', '')))}")
        details_parts.append(f"{escape_markdown_v2(_('major_details_location', location=found_major.get('location', 'N/A')))}")
        desc = found_major.get(f"description_{CURRENT_LANG}", found_major.get("description_en", ""))
        if desc: details_parts.append(f"\n{escape_markdown_v2(_('major_details_description', description=desc))}")
        keywords = found_major.get("keywords", [])
        if keywords: details_parts.append(f"\n{escape_markdown_v2(_('major_details_keywords', keywords=', '.join(keywords)))}")
        careers_list = found_major.get(f"career_prospects_{CURRENT_LANG}", found_major.get("career_prospects_en", []))
        if careers_list: details_parts.append(f"\n{escape_markdown_v2(_('major_details_career_prospects', careers='\\n'.join([f'‐ {c}' for c in careers_list])))}") # Escaping each career prospect might be needed if they contain markdown
        core_subjects_list = found_major.get(f"core_subjects_{CURRENT_LANG}", found_major.get("core_subjects_en", []))
        if core_subjects_list: details_parts.append(f"\n{escape_markdown_v2(_('major_details_core_subjects', core_subjects='\\n'.join([f'‐ {s}' for s in core_subjects_list])))}")
        link = found_major.get("university_major_link")
        if link: details_parts.append(f"\n{_('major_details_link', link=escape_markdown_v2(link))}") # Link itself should be escaped if part of overall markdown
        
        details_text_final = "\n".join(filter(None,details_parts))
        
        reply_markup = create_keyboard([[('back_to_results_button', 'back_to_results')]])
        try: await query.edit_message_text(details_text_final, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
        except Exception as e_md:
            logger.error(f"Markdown error in learn_more: {e_md}. Details: {details_text_final}")
            # Fallback: try sending without Markdown or with minimal Markdown
            plain_text_details = re.sub(r'[*_`~]', '', details_text_final) # Basic un-markdown
            plain_text_details = re.sub(r'\\([_*\[\]()~`>#+\-=|{}.!])', r'\1', plain_text_details) # Remove our own escapes
            await query.edit_message_text(plain_text_details, reply_markup=reply_markup, disable_web_page_preview=True)
    except Exception as e: logger.error(f"Learn more error: {e}", exc_info=True); await query.edit_message_text(_('error_message'))
    return SHOW_DETAILS

async def handle_back_to_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await display_single_recommendation(update, context)
    return SHOW_RESULTS
async def handle_start_over(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await start_command(update, context)
async def handle_end_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); await query.edit_message_text(_('thank_you_message')); context.user_data.clear(); return ConversationHandler.END
async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message: await update.message.reply_text(_('cancel_message'))
    elif update.callback_query: await update.callback_query.answer(); await update.callback_query.edit_message_text(_('cancel_message'))
    context.user_data.clear(); return ConversationHandler.END

def main() -> None:
    if "YOUR_BOT_TOKEN" in TELEGRAM_BOT_TOKEN or "your_bot_username" in BOT_USERNAME: # Basic check
        logger.critical("FATAL: Bot token or username is a placeholder."); print("FATAL: Bot token or username needs to be configured."); return
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(ask_name_handler, pattern="^start_assessment$")],
        states={
            ASK_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_name)],
            ASK_CURRENT_STATUS: [CallbackQueryHandler(handle_current_status, pattern="^status_.*$")],
            ASK_SUBJECT_STRENGTH: [CallbackQueryHandler(handle_subject_strength, pattern="^subject_.*$")],
            ASK_PROBLEM_SOLVING_STYLE: [CallbackQueryHandler(handle_problem_solving_style, pattern="^style_.*$")],
            ASK_WORK_ENVIRONMENT_PREFERENCE: [CallbackQueryHandler(handle_work_environment_preference, pattern="^env_.*$")],
            ASK_IMPACT_ASPIRATION: [CallbackQueryHandler(handle_impact_aspiration, pattern="^impact_.*$")],
            ASK_STUDY_LOCATION: [CallbackQueryHandler(handle_study_location, pattern="^location_.*$")],
            ASK_BUDGET_RANGE: [CallbackQueryHandler(handle_budget_range, pattern="^budget_.*$")],
            ASK_BAC_RESULT: [CallbackQueryHandler(handle_bac_result, pattern="^bac_.*$")],
            ASK_LEARN_STYLE: [CallbackQueryHandler(handle_learn_style, pattern="^learn_.*$")],
            ASK_EXTRA_ACTIVITY: [CallbackQueryHandler(handle_extra_activity, pattern="^activity_.*$")],
            ASK_SCHOLARSHIP: [CallbackQueryHandler(handle_scholarship, pattern="^scholarship_.*$")],
            CONFIRM_INPUTS_AND_PROCESS: [
                CallbackQueryHandler(process_answers_and_recommend, pattern="^process_answers$"),
                # Back button from confirmation goes to the last question asked before confirm: Scholarship
                CallbackQueryHandler(handle_extra_activity, pattern="^back_to_ask_scholarship$")
            ],
            SHOW_RESULTS: [
                CallbackQueryHandler(handle_learn_more, pattern="^learn_more_.*$"),
                CallbackQueryHandler(handle_next_suggestion, pattern="^next_suggestion$"),
                CallbackQueryHandler(handle_start_over, pattern="^start_over$"),
                CallbackQueryHandler(handle_end_conversation, pattern="^end_conversation$"),
            ],
            SHOW_DETAILS: [
                CallbackQueryHandler(handle_back_to_results, pattern="^back_to_results$"),
                CallbackQueryHandler(handle_start_over, pattern="^start_over$"),
                CallbackQueryHandler(handle_end_conversation, pattern="^end_conversation$"),
            ],
            SELECT_ACTION: [
                CallbackQueryHandler(ask_name_handler, pattern="^start_assessment$"),
                CallbackQueryHandler(handle_show_about, pattern="^show_about$"),
                CallbackQueryHandler(start_command, pattern="^main_menu$")
            ]
        },
        fallbacks=[
            CommandHandler("start", start_command), CommandHandler("cancel", cancel_command),
            CallbackQueryHandler(start_command, pattern="^main_menu$"),
            CallbackQueryHandler(ask_name_handler, pattern="^back_to_ask_name$"),
            CallbackQueryHandler(handle_name, pattern="^back_to_ask_current_status$"), # Asks current status
            CallbackQueryHandler(handle_current_status, pattern="^back_to_ask_subject_strength$"), # Asks subject strength
            CallbackQueryHandler(handle_subject_strength, pattern="^back_to_ask_problem_solving_style$"),
            CallbackQueryHandler(handle_problem_solving_style, pattern="^back_to_ask_work_environment_preference$"),
            CallbackQueryHandler(handle_work_environment_preference, pattern="^back_to_ask_impact_aspiration$"),
            CallbackQueryHandler(handle_impact_aspiration, pattern="^back_to_ask_study_location$"),
            CallbackQueryHandler(handle_study_location, pattern="^back_to_ask_budget_range$"),
            CallbackQueryHandler(handle_budget_range, pattern="^back_to_ask_bac_result$"),
            CallbackQueryHandler(handle_bac_result, pattern="^back_to_ask_learn_style$"),
            CallbackQueryHandler(handle_learn_style, pattern="^back_to_ask_extra_activity$"),
            # Fallback for back_to_ask_scholarship is already in CONFIRM_INPUTS_AND_PROCESS state's fallbacks implicitly by pattern
            # Or explicitly:
            # CallbackQueryHandler(handle_extra_activity, pattern="^back_to_ask_scholarship$"), # This is handled in CONFIRM_INPUTS_AND_PROCESS
        ],
        allow_reentry=True,
    )
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(conv_handler)
    app.add_handler(CallbackQueryHandler(start_command, pattern="^main_menu$"))
    logger.info(f"Bot @{BOT_USERNAME} is starting...")
    print(f"Bot @{BOT_USERNAME} is starting. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()