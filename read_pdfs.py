import sys
import subprocess

try:
    import pypdf
except ImportError:
    print("Installing pypdf...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "--user"])
    import pypdf

def extract_pdf_preview(filename):
    try:
        reader = pypdf.PdfReader(filename)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        print(f"--- {filename} ---")
        if not text.strip():
            print("No extractable text or scanned PDF.")
        else:
            print(text[:1500])
        print("=========\n")
    except Exception as e:
         print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    extract_pdf_preview("ABC様_AIデータセンター 開通通知書.pdf")
    extract_pdf_preview("【顧客マニュアル】AIデータセンター GPUサーバー_ご利用ガイド.pdf")
    extract_pdf_preview("【顧客マニュアル】法人テクニカルサポートWebご利用ガイド.pdf")
