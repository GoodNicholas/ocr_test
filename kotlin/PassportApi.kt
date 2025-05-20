@file:JvmName("PassportApi")

import ai.onnxruntime.*
import net.sourceforge.tess4j.ITessAPI.TessPageSegMode
import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB
import org.bytedeco.opencv.global.opencv_imgproc.cvtColor
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.Rect
import java.awt.image.BufferedImage
import org.bytedeco.opencv.opencv_core.Mat
import java.awt.image.DataBufferByte
import org.bytedeco.opencv.opencv_core.*
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.FloatBuffer
import javax.imageio.ImageIO
import kotlin.math.min
import kotlin.collections.maxBy
import kotlin.collections.maxWith


object PassportApi : AutoCloseable {

    private val NAMES = listOf(
        "surname", "name", "patronymic", "gender", "dob",
        "birth_place", "issuing_auth", "issue_date",
        "division_code", "series_number", "mrz"
    )

    private val FIELD_CLASSES = mapOf(
        "surname" to 0,
        "name" to 1,
        "patronymic" to 2,
        "gender" to 3,
        "dob" to 4,
        "birth_place" to 5,
        "issuing_auth" to 6,
        "issue_date" to 7,
        "division_code" to 8,
        "series_number_top" to 9,
        "series_number_bottom" to 9
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

    private val reader = OCRReader.Builder().build()

    private val BBOX_MARGIN = 5

    private const val DEFAULT_MODEL =
        "C:\\Users\\Admin\\Downloads\\best segmenter extra.onnx"

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession =
        env.createSession(DEFAULT_MODEL, OrtSession.SessionOptions())

    private val inputName = session.inputNames.first()
    private val outputName = session.outputNames.first()

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

    val REGEXPS = mapOf(
        "division_code" to Regex("""\d{3}-\d{3}"""),
        "dob"           to Regex("""\d{2}\.\d{2}\.\d{4}"""),
        "issue_date"    to Regex("""\d{2}\.\d{2}\.\d{4}"""),
        "series_number" to Regex("""\d{2} \d{2} \d{6}""")
    )

    fun segmentAndGetBestFields(
        imagePath: String,
        saveCrops: Boolean = false,
        outDir: String = "crops",
        confThres: Float = 0.25f
    ): PassportFields {
        val results = segmentAndReadPassport(imagePath, saveCrops, outDir, confThres)

        // Для удобства: сгруппировать по fieldName
        val byField = results.groupBy { it.fieldName }

        fun bestByRegexOrScore(field: String): String? {
            val regex = REGEXPS[field] ?: return byField[field]?.maxBy { it.score }?.text
            val list = byField[field] ?: return null

            return list
                .map {
                    val value = it.text.orEmpty()
                    val percent = regexMatchPercent(value, regex)
                    Triple(it, percent, it.score)
                }
                .maxWith(
                    compareBy<Triple<FieldData, Double, Float>>(
                        { it.second }, // процент совпадения по регэкспу
                        { it.third }   // score, если одинаковый процент
                    )
                )
                ?.first?.text
        }


        return PassportFields(
            surname        = byField["surname"]?.takeIf { it.isNotEmpty() }?.maxBy { it.score }?.text,
            name           = byField["name"]?.takeIf { it.isNotEmpty() }?.maxBy { it.score }?.text,
            patronymic     = byField["patronymic"]?.takeIf { it.isNotEmpty() }?.maxBy { it.score }?.text,
            gender         = byField["gender"]?.takeIf { it.isNotEmpty() }?.maxBy { it.score }?.text,
            dob            = bestByRegexOrScore("dob"),
            birth_place    = byField["birth_place"]?.takeIf { it.isNotEmpty() }?.maxBy { it.score }?.text,
            issuing_auth   = byField["issuing_auth"]?.takeIf { it.isNotEmpty() }?.maxBy { it.score }?.text,
            issue_date     = bestByRegexOrScore("issue_date"),
            division_code  = bestByRegexOrScore("division_code"),
            series_number  = bestByRegexOrScore("series_number"),
            mrz            = byField["mrz"]?.takeIf { it.isNotEmpty() }?.maxBy { it.score }?.text
        )
    }

    fun segmentAndReadPassport(imagePath: String, saveCrops: Boolean = false, outDir: String = "crops", confThres: Float = 0.25f): List<FieldData> {
        val detections = segmentPassport(imagePath, saveCrops, outDir, confThres)
        val fieldsData = mutableListOf<FieldData>()

        for (detection in detections) {
            val fieldName = detection.fieldName
            if (fieldName == "mrz") continue
            val psmMode = FIELD_PSM_MODES[fieldName] ?: TessPageSegMode.PSM_AUTO

            reader.setPageSegMode(psmMode)

            val x = detection.bbox.x()
            val y = detection.bbox.y()
            val w = detection.bbox.width()
            val h = detection.bbox.height()

            val newX = maxOf(0, x - BBOX_MARGIN)
            val newY = maxOf(0, y - BBOX_MARGIN)
            val newWidth = detection.bbox.width() + BBOX_MARGIN * 2
            val newHeight = detection.bbox.height() + BBOX_MARGIN * 2
            val newBbox = Rect(newX, newY, newWidth, newHeight)

            val src = imread(imagePath) ?: error("Cannot read image at $imagePath")

            val croppedImage = Mat(src, newBbox)
            val readyMat = if (croppedImage.isContinuous) croppedImage else croppedImage.clone()
            val buffered = readyMat.toBufferedImage()
            val text = reader.read(buffered)


            fieldsData.add(
                FieldData(
                    fieldName = fieldName,
                    text = text.takeIf { it.isNotEmpty() },
                    score = detection.score,
                    bbox = detection.bbox
                )
            )
        }

        return fieldsData
    }

    fun Mat.toBufferedImage(): BufferedImage {
        var mat = this
        if (mat.channels() == 1) {
            val tmp = Mat()
            cvtColor(mat, tmp, org.bytedeco.opencv.global.opencv_imgproc.COLOR_GRAY2BGR)
            mat = tmp
        }
        val width = mat.cols()
        val height = mat.rows()
        val channels = mat.channels()
        val sourcePixels = ByteArray(width * height * channels)
        mat.data().get(sourcePixels)
        val image = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
        val targetPixels = (image.raster.dataBuffer as DataBufferByte).data
        System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.size)
        return image
    }

    /**
     * Главная «внешняя» функция API.
     *
     * @param imagePath  путь к изображению паспорта
     * @param saveCrops  при true сохранит кропы в директорию [outDir]
     * @param outDir     каталог для кропов
     * @param confThres  порог уверенности
     * @return список детекций
     */
    fun segmentPassport(
        imagePath: String,
        saveCrops: Boolean = false,
        outDir: String = "crops",
        confThres: Float = 0.25f
    ): List<Detection> {
        val src = imread(imagePath)
            ?: error("Cannot read image at $imagePath")

        val (padded, scale, padX, padY) = letterbox(src)

        val tensor = padded.toCHWTensor(env)

        val result = session.run(mapOf(inputName to tensor))

        val detections = mutableListOf<Detection>()

        try {
            val ortVal = result.get(outputName)
                .orElseThrow { IllegalStateException("No output named `$outputName`") }

            require(ortVal is OnnxTensor) {
                "Expected OnnxTensor, but got ${ortVal::class.java.simpleName}"
            }
            val onnxTensor = ortVal as OnnxTensor

            val raw = onnxTensor.value
            @Suppress("UNCHECKED_CAST")
            val det2d: Array<FloatArray> = when {
                raw is Array<*> && raw.isArrayOf<FloatArray>() ->
                    raw as Array<FloatArray>
                raw is Array<*> && raw.isArrayOf<Array<*>>() ->
                    (raw as Array<Array<FloatArray>>)[0]
                else ->
                    error("Unexpected output shape: ${raw::class}")
            }

            det2d.forEachIndexed { idx, det ->
                val score = det[4]
                if (score < confThres) return@forEachIndexed

                val clsId = det[5].toInt()
                val x1 = ((det[0] - padX) / scale).toInt().coerceIn(0, src.cols())
                val y1 = ((det[1] - padY) / scale).toInt().coerceIn(0, src.rows())
                val x2 = ((det[2] - padX) / scale).toInt().coerceIn(0, src.cols())
                val y2 = ((det[3] - padY) / scale).toInt().coerceIn(0, src.rows())
                val rect = Rect(x1, y1, x2 - x1, y2 - y1)

                if (saveCrops) {
                    val clsDir = File(outDir, "cls_$clsId").apply { mkdirs() }
                    val fn = "${File(imagePath).nameWithoutExtension}_cls${clsId}_$idx.jpg"
                    imwrite(File(clsDir, fn).absolutePath, Mat(src, rect))
                }

                detections += Detection(
                    fieldName = NAMES[clsId],
                    score     = score,
                    bbox      = rect
                )
            }
        } finally {
            result.close()
            tensor.close()
        }

        return detections
    }

    private data class LetterboxResult(
        val img: Mat,
        val scale: Float,
        val padX: Int,
        val padY: Int
    )

    private fun letterbox(src: Mat, size: Int = 640): LetterboxResult {
        val (h, w) = src.rows() to src.cols()
        val scale = min(size.toFloat() / h, size.toFloat() / w)
        val nh = (h * scale).toInt()
        val nw = (w * scale).toInt()

        val resized = Mat()
        resize(src, resized, Size(nw, nh))

        val top = (size - nh) / 2
        val bottom = size - nh - top
        val left = (size - nw) / 2
        val right = size - nw - left

        val padded = Mat()
        copyMakeBorder(
            resized, padded, top, bottom, left, right,
            BORDER_CONSTANT, Scalar(114.0, 114.0, 114.0, 0.0)
        )
        return LetterboxResult(padded, scale, left, top)
    }

    private fun Mat.toCHWTensor(env: OrtEnvironment): OnnxTensor {
        val floatMat = Mat()
        this.convertTo(floatMat, CV_32FC3, 1.0 / 255.0, 0.0)

        val n = floatMat.rows() * floatMat.cols() * 3
        val dataPtr = FloatPointer(floatMat.data())
        val srcArray = FloatArray(n).also { dataPtr.get(it) }

        val buffer = FloatBuffer.allocate(n)
        for (c in 2 downTo 0) {          // BGR ➜ RGB
            for (i in 0 until floatMat.rows() * floatMat.cols()) {
                buffer.put(srcArray[i * 3 + c])
            }
        }
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, longArrayOf(1, 3, 640, 640))
    }

    fun regexMatchPercent(value: String, regex: Regex): Double {
        val match = regex.find(value) ?: return 0.0
        return match.value.length.toDouble() / regex.pattern.length
    }

    override fun close() {
        session.close()
        env.close()
    }
}
