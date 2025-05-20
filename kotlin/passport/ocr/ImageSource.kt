package passport.ocr

import java.awt.image.BufferedImage
import java.io.InputStream

sealed class ImageSource {
    data class Path(val path: String) : ImageSource()
    data class ByteArraySource(val data: ByteArray) : ImageSource()
    data class Stream(val stream: InputStream) : ImageSource()
    data class Buffered(val image: BufferedImage) : ImageSource()
}