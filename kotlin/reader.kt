package com.example.ocr

import net.sourceforge.tess4j.ITessAPI.TessOcrEngineMode
import net.sourceforge.tess4j.ITessAPI.TessPageSegMode
import net.sourceforge.tess4j.Tesseract
import java.awt.image.BufferedImage
import java.io.File
import java.io.InputStream

sealed class ImageSource {
    data class Path(val path: String) : ImageSource()
    data class ByteArraySource(val data: ByteArray) : ImageSource()
    data class Stream(val stream: InputStream) : ImageSource()
    data class Buffered(val image: BufferedImage) : ImageSource()
}

class OCRReader private constructor(
    private val tessDataPath: String,
    private val language: String,
    private val oem: Int,
    private val psm: Int
) {
    private val tesseract = Tesseract().apply {
        setDatapath(tessDataPath)
        setLanguage(language)
        setOcrEngineMode(oem)
        setPageSegMode(psm)
        setTessVariable("user_defined_dpi", "300")
    }

    fun read(source: ImageSource): String = when (source) {
        is ImageSource.Path ->
            tesseract.doOCR(File(source.path))
        is ImageSource.ByteArraySource ->
            tesseract.doOCR(javax.imageio.ImageIO.read(source.data.inputStream()))
        is ImageSource.Stream ->
            tesseract.doOCR(javax.imageio.ImageIO.read(source.stream))
        is ImageSource.Buffered ->
            tesseract.doOCR(source.image)
    }

    fun read(path: String): String =
        read(ImageSource.Path(path))

    fun read(data: ByteArray): String =
        read(ImageSource.ByteArraySource(data))

    fun read(stream: InputStream): String =
        read(ImageSource.Stream(stream))

    fun read(image: BufferedImage): String =
        read(ImageSource.Buffered(image))


    class Builder {
        private var tessDataPath: String = "C:\\Program Files\\Tesseract-OCR\\tessdata"
        private var language: String = "rus"
        private var oem: Int = TessOcrEngineMode.OEM_DEFAULT
        private var psm: Int = TessPageSegMode.PSM_SINGLE_WORD

        fun setTessDataPath(path: String) = apply { this.tessDataPath = path }
        fun setLanguage(lang: String) = apply { this.language = lang }
        fun setOcrEngineMode(oem: Int) = apply { this.oem = oem }
        fun setPageSegMode(psm: Int) = apply { this.psm = psm }
        fun build() = OCRReader(tessDataPath, language, oem, psm)
    }
}
