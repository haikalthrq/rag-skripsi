"""
Modul untuk pembersihan teks menggunakan regex.
Membersihkan header/footer, nomor halaman, whitespace berlebih, karakter aneh, dan pola tidak relevan lainnya.
"""

import re
import logging
from typing import Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Membersihkan teks hasil ekstraksi PDF menggunakan regex.
    
    Proses pembersihan meliputi:
    - Menghapus nomor halaman
    - Menghapus header dan footer umum
    - Menghapus whitespace berlebih
    - Menghapus karakter khusus yang tidak relevan
    - Menghapus pola repetitif
    - Normalisasi spasi dan line breaks
    
    Args:
        text (str): Teks yang akan dibersihkan.
        
    Returns:
        str: Teks yang telah dibersihkan.
    """
    if not text or not isinstance(text, str):
        logger.warning("Input text kosong atau bukan string")
        return ""
    
    original_length = len(text)
    cleaned = text
    
    # 1. Hapus byte order marks dan karakter zero-width
    cleaned = re.sub(r'[\ufeff\u200b\u200c\u200d]', '', cleaned)
    
    # 2. Hapus nomor halaman yang umum
    # Pola: "Page 1", "Halaman 1", "1 of 10", "[1]", dll
    cleaned = re.sub(r'\b[Pp]age\s+\d+\b', '', cleaned)
    cleaned = re.sub(r'\b[Hh]alaman\s+\d+\b', '', cleaned)
    cleaned = re.sub(r'\b\d+\s+of\s+\d+\b', '', cleaned)
    cleaned = re.sub(r'^\s*\[\d+\]\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)
    
    # 3. Hapus header/footer yang berulang
    # Pola umum: garis horizontal, copyright, URL, email
    cleaned = re.sub(r'^[\s\-_=]{3,}$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'©\s*\d{4}.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\b[Cc]opyright\s*©?\s*\d{4}.*$', '', cleaned, flags=re.MULTILINE)
    
    # 4. Hapus URL dan email
    cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
    cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned)
    
    # 5. Hapus karakter kontrol dan karakter khusus yang tidak perlu
    # Pertahankan huruf, angka, spasi, dan tanda baca standar
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
    
    # 6. Hapus karakter Unicode yang tidak umum (optional, sesuaikan dengan kebutuhan)
    # Pertahankan huruf Latin, angka, dan tanda baca umum
    # cleaned = re.sub(r'[^\x20-\x7E\u00A0-\u024F\u1E00-\u1EFF]', '', cleaned)
    
    # 7. Hapus bullet points dan simbol list yang berlebih
    cleaned = re.sub(r'^[\s•\-\*\+●○■□▪▫]+', '', cleaned, flags=re.MULTILINE)
    
    # 8. Hapus referensi footnote/endnote
    cleaned = re.sub(r'\[\d+\]', '', cleaned)
    cleaned = re.sub(r'\(\d+\)', '', cleaned)
    
    # 9. Normalisasi whitespace
    # Hapus spasi di awal dan akhir baris
    cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^[ \t]+', '', cleaned, flags=re.MULTILINE)
    
    # Hapus multiple spaces menjadi single space
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    # Hapus multiple tabs menjadi single space
    cleaned = re.sub(r'\t+', ' ', cleaned)
    
    # 10. Normalisasi line breaks
    # Hapus 3+ newlines berturut-turut menjadi 2 newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Hapus newlines di tengah kalimat (line breaks yang tidak perlu)
    # Hati-hati agar tidak menggabungkan paragraf yang berbeda
    cleaned = re.sub(r'(?<=[a-z,;])\n(?=[a-z])', ' ', cleaned)
    
    # 11. Hapus baris kosong yang hanya berisi whitespace
    cleaned = re.sub(r'^\s*$\n', '', cleaned, flags=re.MULTILINE)
    
    # 12. Hapus pola repetitif (misalnya "..." atau "---")
    cleaned = re.sub(r'\.{3,}', '...', cleaned)  # Normalisasi ellipsis
    cleaned = re.sub(r'-{3,}', '', cleaned)  # Hapus garis panjang
    cleaned = re.sub(r'_{3,}', '', cleaned)
    cleaned = re.sub(r'={3,}', '', cleaned)
    
    # 13. Trim awal dan akhir
    cleaned = cleaned.strip()
    
    # Log hasil pembersihan
    cleaned_length = len(cleaned)
    reduction_percent = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
    
    logger.info(f"Pembersihan teks selesai:")
    logger.info(f"  - Karakter awal: {original_length}")
    logger.info(f"  - Karakter akhir: {cleaned_length}")
    logger.info(f"  - Pengurangan: {reduction_percent:.2f}%")
    
    return cleaned


def clean_text_advanced(text: str, remove_numbers: bool = False, 
                       remove_punctuation: bool = False) -> str:
    """
    Pembersihan teks dengan opsi tambahan.
    
    Args:
        text (str): Teks yang akan dibersihkan.
        remove_numbers (bool): Jika True, hapus semua angka.
        remove_punctuation (bool): Jika True, hapus semua tanda baca.
        
    Returns:
        str: Teks yang telah dibersihkan.
    """
    # Lakukan pembersihan standar terlebih dahulu
    cleaned = clean_text(text)
    
    if remove_numbers:
        # Hapus semua angka
        cleaned = re.sub(r'\d+', '', cleaned)
    
    if remove_punctuation:
        # Hapus tanda baca, pertahankan spasi
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
    
    # Normalisasi ulang whitespace setelah penghapusan tambahan
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def remove_headers_footers(text: str, pattern_list: Optional[list] = None) -> str:
    """
    Menghapus header dan footer berdasarkan pola yang diberikan.
    
    Args:
        text (str): Teks yang akan diproses.
        pattern_list (list): List regex pattern untuk header/footer yang akan dihapus.
        
    Returns:
        str: Teks dengan header/footer yang sudah dihapus.
    """
    if pattern_list is None:
        # Pola default untuk header/footer umum
        pattern_list = [
            r'^\s*Badan Pusat Statistik.*$',
            r'^\s*Statistics Indonesia.*$',
            r'^\s*www\.bps\.go\.id.*$',
            r'^\s*Katalog.*:.*$',
            r'^\s*ISSN.*:.*$',
            r'^\s*ISBN.*:.*$',
        ]
    
    cleaned = text
    for pattern in pattern_list:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Normalisasi whitespace setelah penghapusan
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


if __name__ == "__main__":
    # Testing
    test_text = """
    Page 1
    
    Badan Pusat Statistik
    www.bps.go.id
    
    Halaman 1 of 10
    
    Ini adalah contoh teks     dengan spasi berlebih.
    
    
    
    Dan juga newline      yang terlalu banyak.
    
    URL: https://example.com dan email test@example.com
    
    © 2024 Copyright Notice
    
    ---------------------------------
    
    [1] Footnote reference
    
    ..................
    
    Teks normal di sini.
    """
    
    print("=== Teks Asli ===")
    print(test_text)
    print(f"\nPanjang: {len(test_text)} karakter")
    
    print("\n" + "="*50)
    print("=== Teks Setelah Dibersihkan ===")
    cleaned = clean_text(test_text)
    print(cleaned)
    print(f"\nPanjang: {len(cleaned)} karakter")
