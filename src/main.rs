use serde_json::{Value, json};
use rig::embeddings::{Embedding, EmbeddingModel, EmbeddingsBuilder};
use rig_fastembed::{Client, FastembedModel};
use anyhow::Result;

#[tokio::main]
async fn main() {
    let core_text = std::fs::read_to_string("example.txt")
        .expect("Failed to read example.txt");

    let fs2i_json = convert_to_fs2i(&core_text).await;

    let output = serde_json::to_string(&fs2i_json)
        .expect("Failed to convert to JSON string");
    println!("{}", output);
}

async fn convert_to_fs2i(text: &str) -> Value {
    json!({
        "str_len": text.len(),
        "str_words": text.split_whitespace().count(),
        "chunks": chunk_text(text).await,
        "converted": true
    })
}

async fn chunk_text(text: &str) -> Vec<Value> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let max_chunk_size = 1200;
    let tolerance = 200; // on autorise +200 ou -200 autour de 1200

    while start < text.len() {
        let remaining = &text[start..];

        if remaining.len() <= max_chunk_size + tolerance {
            chunks.push(Value::String(remaining.trim().to_string()));
            break;
        }

        let search_area = &remaining[..(max_chunk_size + tolerance).min(remaining.len())];

        // Cherche le point le plus proche autour de 1000
        let mut split_at = search_area[..max_chunk_size]
            .rfind('.')
            .unwrap_or_else(|| {
                // si aucun point avant, cherche après
                search_area[max_chunk_size..]
                    .find('.')
                    .map(|i| i + max_chunk_size)
                    .unwrap_or(max_chunk_size) // sinon coupe à 1000 sec
            });

        // Avance après le point final trouvé
        split_at += 1;

        let chunk = &remaining[..split_at];
        chunks.push(Value::String(chunk.trim().to_string()));
        start += split_at;
    }

    let embeds = vectorize_text_simple(chunks.iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect()).await;

    let mut all_chunks:Vec<Value> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_str = chunk.as_str().unwrap_or("");
        let chunk_len = chunk_str.len();
        let chunk_words = chunk_str.split_whitespace().count();

        // Create a JSON object for each chunk
        let chunk_json = json!({
            "chunk": chunk,
            "chunk_len": chunk_len,
            "chunk_words": chunk_words,
            "vector": embeds.get(i).cloned().unwrap_or_default()
        });

        all_chunks.push(chunk_json);
    }

    std::fs::write("tmp.json", serde_json::to_string(&all_chunks).unwrap())
        .expect("Failed to write chunks to file");

    chunks
}


pub async fn vectorize_text_simple(texts: Vec<String>) -> Vec<Vec<String>> {
    let client = Client::new();

    // Récupération du modèle dyn
    let model = client.embedding_model(&FastembedModel::AllMiniLML6V2Q);

    // Appel direct sur le trait dyn EmbeddingModel
    let embeddings_vec = model.embed_texts(texts).await; // ✅ méthode correcte ici
    let embeddings_vec = embeddings_vec.unwrap_or_else(|_| vec![]);

    return embeddings_vec
        .into_iter()
        .map(|e| e.vec.into_iter().map(|v| v.to_string()).collect())
        .collect();
}