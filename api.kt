@file:JvmName("PassportApi")

import ai.onnxruntime.*
import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.*
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.min

object PassportApi : AutoCloseable {

    private val NAMES = listOf(
        "surname", "name", "patronymic", "gender", "dob",
        "birth_place", "issuing_auth", "issue_date",
        "division_code", "series_number", "mrz"
    )

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

    override fun close() {
        session.close()
        env.close()
    }
}
