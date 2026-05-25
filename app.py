import io
import json
import os
import re
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
import fitz  # PyMuPDF
from openai import OpenAI

app = Flask(__name__)

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def pdf_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def book_cache_dir(book_id: str) -> Path:
    d = CACHE_DIR / book_id
    d.mkdir(exist_ok=True)
    return d


def chunk_text(text: str, max_chars: int = 3500) -> list[str]:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks, current = [], ''
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current += (' ' if current else '') + sentence
        else:
            if current:
                chunks.append(current.strip())
            if len(sentence) > max_chars:
                words = sentence.split()
                current = ''
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        current += (' ' if current else '') + word
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = word
            else:
                current = sentence
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c.strip()]


SKIP_TITLES = {'index', 'bibliography', 'references', 'table of contents',
               'contents', 'acknowledgements', 'acknowledgments', 'copyright',
               'about the author', 'colophon', 'glossary'}

def should_skip(title: str) -> bool:
    return title.strip().lower() in SKIP_TITLES


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Receive PDF, extract TOC and page count. Returns book_id + chapters."""
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file'}), 400

    pdf_bytes = request.files['pdf'].read()
    book_id = pdf_hash(pdf_bytes)
    title = request.files['pdf'].filename.replace('.pdf', '').replace('.PDF', '')

    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        toc = doc.get_toc()  # [[level, title, page], ...]

        chapters = []
        if toc:
            for i, (level, ch_title, page) in enumerate(toc):
                next_page = toc[i + 1][2] if i + 1 < len(toc) else len(doc)
                chapters.append({
                    'id': i,
                    'level': level,
                    'title': ch_title,
                    'start_page': page - 1,      # 0-indexed
                    'end_page': next_page - 1,   # exclusive
                    'skip': should_skip(ch_title),
                })
        else:
            # No TOC — treat whole book as one chapter
            chapters.append({
                'id': 0,
                'level': 1,
                'title': 'Full Book',
                'start_page': 0,
                'end_page': len(doc),
                'skip': False,
            })

        total_pages = len(doc)
        doc.close()

        # Save PDF bytes to cache for later extraction
        book_dir = book_cache_dir(book_id)
        pdf_path = book_dir / 'source.pdf'
        if not pdf_path.exists():
            pdf_path.write_bytes(pdf_bytes)

        # Save metadata
        meta = {'title': title, 'chapters': chapters, 'pages': total_pages}
        (book_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')

        return jsonify({'book_id': book_id, 'title': title,
                        'chapters': chapters, 'pages': total_pages})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/extract', methods=['POST'])
def extract():
    """Extract and chunk text for selected chapters of a book."""
    try:
        data = request.json
        book_id = data.get('book_id')
        selected_ids = set(data.get('chapter_ids', []))

        book_dir = book_cache_dir(book_id)
        meta_path = book_dir / 'meta.json'
        pdf_path  = book_dir / 'source.pdf'

        if not meta_path.exists() or not pdf_path.exists():
            return jsonify({'error': 'Book data missing — please re-upload the PDF.'}), 400

        meta      = json.loads(meta_path.read_text(encoding='utf-8'))
        chapters  = meta['chapters']
        pdf_bytes = pdf_path.read_bytes()

        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        chunks = []
        for ch in chapters:
            if ch['id'] not in selected_ids:
                continue
            text = ''
            for p in range(ch['start_page'], min(ch['end_page'], len(doc))):
                text += doc[p].get_text()
            for c in chunk_text(text):
                chunks.append({'chapter': ch['title'], 'text': c})
        doc.close()

        (book_dir / 'chunks.json').write_text(
            json.dumps(chunks, ensure_ascii=False), encoding='utf-8')

        return jsonify({'chunks': chunks, 'total': len(chunks)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tts', methods=['POST'])
def tts():
    """Convert one chunk to audio. Checks disk cache first."""
    data = request.json
    api_key = (data.get('api_key') or '').strip()
    text = (data.get('text') or '').strip()
    voice = data.get('voice', 'nova')
    book_id = data.get('book_id', '')
    chunk_index = data.get('chunk_index', -1)

    if not api_key:
        return jsonify({'error': 'API key required'}), 400
    if not text:
        return jsonify({'error': 'Text required'}), 400

    # Check disk cache
    if book_id and chunk_index >= 0:
        cache_file = book_cache_dir(book_id) / f'chunk_{chunk_index:05d}_{voice}.mp3'
        if cache_file.exists():
            return send_file(cache_file, mimetype='audio/mpeg')

    try:
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model='tts-1', voice=voice, input=text)

        audio_bytes = response.content

        # Save to disk cache
        if book_id and chunk_index >= 0:
            cache_file = book_cache_dir(book_id) / f'chunk_{chunk_index:05d}_{voice}.mp3'
            cache_file.write_bytes(audio_bytes)

        return send_file(io.BytesIO(audio_bytes), mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-full', methods=['POST'])
def download_full():
    """Concatenate all cached chunks for a book and return as one MP3."""
    data = request.json
    book_id = data.get('book_id', '')
    voice = data.get('voice', 'nova')

    book_dir = book_cache_dir(book_id)
    meta = json.loads((book_dir / 'meta.json').read_text(encoding='utf-8'))

    chunk_files = sorted(book_dir.glob(f'chunk_*_{voice}.mp3'))
    if not chunk_files:
        return jsonify({'error': 'No cached audio found. Listen to the book first.'}), 400

    combined = io.BytesIO()
    for f in chunk_files:
        combined.write(f.read_bytes())
    combined.seek(0)

    filename = re.sub(r'[^\w\s-]', '', meta['title'])[:60] + '.mp3'
    return send_file(combined, mimetype='audio/mpeg',
                     as_attachment=True, download_name=filename)


@app.route('/library', methods=['GET'])
def library():
    """List all previously converted books."""
    books = []
    for book_dir in CACHE_DIR.iterdir():
        meta_path = book_dir / 'meta.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        mp3_files = list(book_dir.glob('chunk_*.mp3'))
        total_chunks = len(json.loads((book_dir / 'chunks.json').read_text(encoding='utf-8'))) \
            if (book_dir / 'chunks.json').exists() else 0
        books.append({
            'book_id': book_dir.name,
            'title': meta['title'],
            'converted': len(mp3_files),
            'total': total_chunks,
        })
    books.sort(key=lambda b: b['title'])
    return jsonify({'books': books})


@app.route('/library/<book_id>', methods=['DELETE'])
def delete_book(book_id):
    """Remove a book from the library."""
    import shutil
    book_dir = CACHE_DIR / book_id
    if book_dir.exists():
        shutil.rmtree(book_dir)
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
