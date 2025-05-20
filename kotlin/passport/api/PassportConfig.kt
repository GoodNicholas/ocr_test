package passport.api

import net.sourceforge.tess4j.ITessAPI.TessPageSegMode

val FIELD_NAMES = listOf(
    "surname", "name", "patronymic", "gender", "dob",
    "birth_place", "issuing_auth", "issue_date",
    "division_code", "series_number", "mrz"
)

val FIELD_PSM_MODES = mapOf(
    "surname" to TessPageSegMode.PSM_SINGLE_WORD,
    "name" to TessPageSegMode.PSM_SINGLE_WORD,
    "patronymic" to TessPageSegMode.PSM_SINGLE_WORD,
    "gender" to TessPageSegMode.PSM_SINGLE_WORD,
    "dob" to TessPageSegMode.PSM_SINGLE_LINE,
    "birth_place" to TessPageSegMode.PSM_SINGLE_LINE,
    "issuing_auth" to TessPageSegMode.PSM_SINGLE_BLOCK,
    "issue_date" to TessPageSegMode.PSM_SINGLE_LINE,
    "division_code" to TessPageSegMode.PSM_SINGLE_LINE,
    "series_number_top" to TessPageSegMode.PSM_SINGLE_BLOCK_VERT_TEXT,
    "series_number_bottom" to TessPageSegMode.PSM_SINGLE_BLOCK_VERT_TEXT
)

val REGEXPS = mapOf(
    "division_code" to Regex("""\d{3}-\d{3}"""),
    "dob"           to Regex("""\d{2}\.\d{2}\.\d{4}"""),
    "issue_date"    to Regex("""\d{2}\.\d{2}\.\d{4}"""),
    "series_number" to Regex("""\d{2} \d{2} \d{6}""")
)

// Другие константы
const val BBOX_MARGIN = 5
