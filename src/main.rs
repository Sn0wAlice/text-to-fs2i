use serde_json::{Value, json};
use rig::embeddings::{EmbeddingModel};
use rig_fastembed::{Client, FastembedModel};
use whatlang::{detect, Lang as WhatLang};
use lingua::{Language, LanguageDetectorBuilder};
use std::collections::HashMap;
use stopwords::{Spark, Stopwords};

#[tokio::main]
async fn main() {
    let core_text = std::fs::read_to_string("example.txt")
        .expect("Failed to read example.txt");

    let fs2i_json = convert_to_fs2i(&core_text).await;

    let output = serde_json::to_string(&fs2i_json)
        .expect("Failed to convert to JSON string");

    std::fs::write("./fs2i.json", output)
        .expect("Failed to write fs2i.json");
}

//
//   ___     ___ _
//  |  _|___|_  |_|
//  |  _|_ -|  _| |
//  |_| |___|___|_|
//
async fn convert_to_fs2i(text: &str) -> Value {
    // curent timestamp in ms
    let current_timestamp = chrono::Utc::now().timestamp_millis();

    let lang = detect_language(text)
        .unwrap_or_else(|| "unknown".to_string());
    let (chunks, array_of_word_maps) = chunk_text(text, &lang).await;
    let j = json!({
        "str_len": text.len(),
        "str_words": text.split_whitespace().count(),
        "chunks": chunks,
        "words_map": array_of_word_maps,
        "language": lang,
        "converted": true
    });
    println!("-> Converted text to FS2I format in {} milliseconds", chrono::Utc::now().timestamp_millis() - current_timestamp);
    j
}

async fn chunk_text(text: &str, lang: &str) -> (Vec<Value>, Vec<HashMap<&'static str, Value>>) {
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
        let word_purged = remove_stopwords(chunk_str, lang);

        // Create a JSON object for each chunk
        let chunk_json = json!({
            "chunk": chunk,
            "chunk_len": chunk_len,
            "chunk_words_count": chunk_words,
            "chunk_words_list_count": word_purged.len(),
            "chunk_words_list": word_purged,
            "vector": embeds.get(i).cloned().unwrap_or_default()
        });

        all_chunks.push(chunk_json);
    }

    let mut v = Vec::new();

    for chunk in &all_chunks {
        v.push(chunk.get("chunk_words_list")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().map(|item| {
                    item.as_object().map(|obj| {
                        let v = obj.get("v").cloned().unwrap_or(Value::Null);
                        let q = obj.get("q").cloned().unwrap_or(Value::Number(0.into()));
                        json!({"v": v, "q": q})
                    }).unwrap_or(Value::Null)
                }).collect::<Vec<Value>>()
            }).unwrap_or_default());
    }

    let array_of_word_maps = merge_chunk_word_mapping(v);

    (all_chunks, array_of_word_maps)
}

pub async fn vectorize_text_simple(texts: Vec<String>) -> Vec<Vec<f32>> {
    let client = Client::new();

    // Récupération du modèle dyn
    let model = client.embedding_model(&FastembedModel::AllMiniLML6V2Q);

    // Appel direct sur le trait dyn EmbeddingModel
    let embeddings_vec = model.embed_texts(texts).await; // ✅ méthode correcte ici
    let embeddings_vec = embeddings_vec.unwrap_or_else(|_| vec![]);

    embeddings_vec
        .into_iter()
        .map(|e| e.vec.into_iter().map(|v| v as f32).collect())
        .collect()
}



//
//   __                       _       ___ ___
//  |  |   ___ ___ ___    ___| |_ _ _|  _|  _|
//  |  |__| .'|   | . |  |_ -|  _| | |  _|  _|
//  |_____|__,|_|_|_  |  |___|_| |___|_| |_|
//                |___|

/// Detects the language of the given text.
/// Uses `whatlang` for speed, and `lingua` as a fallback if confidence is low.
pub fn detect_language(text: &str) -> Option<String> {
    let iso_map = get_iso_codes();

    // Try with whatlang
    if let Some(info) = detect(text) {
        if info.confidence() > 0.8 {
            let code = whatlang_to_iso(&info.lang(), &iso_map);
            if let Some(iso) = code {
                return Some(iso);
            }
        }
    }

    // Fallback with lingua
    let languages = vec![
        Language::English,
        Language::French,
        Language::German,
        Language::Spanish,
        Language::Italian,
        Language::Portuguese,
        Language::Greek,
        Language::Dutch,
        Language::Russian,
        Language::Arabic,
        Language::Japanese,
        Language::Korean
    ];

    let detector = LanguageDetectorBuilder::from_languages(&languages).build();
    if let Some(lang) = detector.detect_language_of(text) {
        lingua_to_iso(&lang)
    } else {
        None
    }
}

fn whatlang_to_iso(lang: &WhatLang, map: &HashMap<WhatLang, &str>) -> Option<String> {
    map.get(lang).map(|&s| s.to_string())
}

fn get_iso_codes() -> HashMap<WhatLang, &'static str> {
    HashMap::from([
        (whatlang::Lang::Eng, "en"),
        (whatlang::Lang::Fra, "fr"),
        (whatlang::Lang::Deu, "de"),
        (whatlang::Lang::Spa, "es"),
        (whatlang::Lang::Ita, "it"),
        (whatlang::Lang::Por, "pt"),
        (whatlang::Lang::Ell, "el"), // Greek
        (whatlang::Lang::Nld, "nl"),
        (whatlang::Lang::Rus, "ru"),
        (whatlang::Lang::Ara, "ar"),
        (whatlang::Lang::Jpn, "ja"),
        (whatlang::Lang::Kor, "ko"),
    ])
}

fn lingua_to_iso(lang: &Language) -> Option<String> {
    use lingua::Language::*;
    let code = match lang {
        English => "en",
        French => "fr",
        German => "de",
        Spanish => "es",
        Italian => "it",
        Portuguese => "pt",
        Greek => "el",
        Dutch => "nl",
        Russian => "ru",
        Arabic => "ar",
        Japanese => "ja",
        Korean => "ko",
        _ => return None,
    };
    Some(code.to_string())
}


//
//   _____ _              _ _ _           _
//  |   __| |_ ___ ___   | | | |___ ___ _| |___
//  |__   |  _| . | . |  | | | | . |  _| . |_ -|
//  |_____|_| |___|  _|  |_____|___|_| |___|___|
//                |_|
/// Removes stop words from a given text using the provided ISO 639-1 language code ("en", "fr", etc.).
pub fn remove_stopwords(text: &str, lang_code: &str) -> Vec<HashMap<&'static str, serde_json::Value>> {
    let lang_opt = match lang_code {
        "en" => Some(stopwords::Language::English),
        "fr" => Some(stopwords::Language::French),
        "de" => Some(stopwords::Language::German),
        "es" => Some(stopwords::Language::Spanish),
        "it" => Some(stopwords::Language::Italian),
        "pt" => Some(stopwords::Language::Portuguese),
        "el" => Some(stopwords::Language::Greek),
        "nl" => Some(stopwords::Language::Dutch),
        "ru" => Some(stopwords::Language::Russian),
        "ar" => Some(stopwords::Language::Arabic),
        _ => None,
    };

    if let Some(lang) = lang_opt {
        if let Some(stopwords) = Spark::stopwords(lang) {
            let filtered: Vec<String> = text
                .split_whitespace()
                .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .filter(|word| !stopwords.contains(&&**word))
                .collect();

            // remove duplicates
            return count_values(filtered);
        }
    }

    // If language not supported or missing stopwords, return original text
    count_values(
        text.split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .collect(),
    )
}


//
//   _ _ _           _    _____             _
//  | | | |___ ___ _| |  |     |___ ___ ___|_|___ ___
//  | | | | . |  _| . |  | | | | .'| . | . | |   | . |
//  |_____|___|_| |___|  |_|_|_|__,|  _|  _|_|_|_|_  |
//                                 |_| |_|       |___|
fn count_values<T: std::hash::Hash + Eq + Clone + std::fmt::Debug + serde::ser::Serialize>(filtered: Vec<T>) -> Vec<HashMap<&'static str, serde_json::Value>> {
    let mut counts = HashMap::new();

    for item in filtered {
        *counts.entry(item).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .map(|(value, count)| {
            let mut map = HashMap::new();
            map.insert("v", serde_json::json!(value));
            map.insert("q", serde_json::json!(count));
            map
        })
        .collect()
}

fn merge_chunk_word_mapping(
    chunks: Vec<Vec<Value>>,
) -> Vec<HashMap<&'static str, Value>> {
    let mut global_counts: HashMap<Value, i64> = HashMap::new();

    for chunk in chunks {
        for map in chunk {
            if let (Some(v), Some(q)) = (map.get("v"), map.get("q")) {
                if let (Value::String(value_str), Value::Number(count)) = (v.clone(), q) {
                    if let Some(q_int) = count.as_i64() {
                        *global_counts.entry(Value::String(value_str)).or_insert(0) += q_int;
                    }
                } else if let (Value::Number(value_num), Value::Number(count)) = (v.clone(), q) {
                    if let (Some(v_num), Some(q_int)) = (value_num.as_i64(), count.as_i64()) {
                        *global_counts.entry(Value::Number(v_num.into())).or_insert(0) += q_int;
                    }
                } else {
                    // support for other serde_json::Value types (optional)
                    *global_counts.entry(v.clone()).or_insert(0) += q.as_i64().unwrap_or(1);
                }
            }
        }
    }

    // Convert back to Vec<HashMap<...>>
    global_counts
        .into_iter()
        .map(|(v, q)| {
            let mut map = HashMap::new();
            map.insert("v", v);
            map.insert("q", Value::Number(q.into()));
            map
        })
        .collect()
}

