

fun main() {
    val imagePath = "Z:\\1551114833_Qkr8HprHdtA.jpg"
    val passport = PassportApi.segmentAndGetBestFields(imagePath)

    println("===== Результаты распознавания паспорта =====")
    println("Фамилия         : ${passport.surname ?: "—"}")
    println("Имя             : ${passport.name ?: "—"}")
    println("Отчество        : ${passport.patronymic ?: "—"}")
    println("Пол             : ${passport.gender ?: "—"}")
    println("Дата рождения   : ${passport.dob ?: "—"}")
    println("Место рождения  : ${passport.birth_place ?: "—"}")
    println("Кем выдан       : ${passport.issuing_auth ?: "—"}")
    println("Дата выдачи     : ${passport.issue_date ?: "—"}")
    println("Код подразделения: ${passport.division_code ?: "—"}")
    println("Серия/номер     : ${passport.series_number ?: "—"}")
    println("MRZ             : ${passport.mrz ?: "—"}")
    println("=============================================")
}

