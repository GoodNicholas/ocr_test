package passport.model

import org.bytedeco.opencv.opencv_core.Rect

data class Detection(
    val fieldName: String,
    val score: Float,
    val bbox: Rect
)

data class FieldData(
    val fieldName: String,
    val text: String?,
    val score: Float,
    val bbox: Rect
)

data class PassportFields(
    val surname: String? = null,
    val name: String? = null,
    val patronymic: String? = null,
    val gender: String? = null,
    val dob: String? = null,
    val birth_place: String? = null,
    val issuing_auth: String? = null,
    val issue_date: String? = null,
    val division_code: String? = null,
    val series_number: String? = null,
    val mrz: String? = null
)