# text-to-fs2i
Convert text to format "Full search item interface" (i have no idea what it is, but i like the name, sounds like a good name for a full-text search index item).

## What is this?
This is a Rust POC for converting text files into a format that can be used for full-text search indexing. It reads a text file, splits it into chunks, and generates metadata for each chunk, including its length, word count, and a vector representation.

```
(input) -> text file
   |
   v
[ text-to-fs2i ]
   |
   v
(output) -> JSON file with chunks and metadata
```
### Output format
```
{
  "chunks": [
    {
      "chunk": "TEXT_HERE",
      "chunk_len": +-1200,
      "chunk_words_count": XX,
      "chunk_words_list_count": XX,
      "chunk_words_list": [ // list of all words without stop words and count
        {"v": "word1", "q": 2},
        {"v": "word2", "q": 3},
        // ...
      ],
      "vector": [
       // list of points of vector with full text
      ]
    },
    
    {/*  OTHER CHUNK  */}
  ],
  "converted": true, // <- allways true, it's a dev watchdog
  "language": "fr", // detected lang (default 'unknow')
  "str_len": XXXX, // total length
  "str_words": XXXX // total worlds
  "words_map": [] // merged wordmap
}
```

# Usage
> use me as example, copy past me in other rust projects


## Exemple input/output

- [input text](./example/input.txt)
- [output json](./example/output.json)