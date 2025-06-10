//package ai.mlc.mlcchat
//
//import android.content.Context
//import android.net.Uri
//import android.util.Log
//import ai.mlc.mlcllm.MLCEngine
//import kotlin.math.sqrt
//import ai.mlc.mlcllm.getEmbedding
//import ai.mlc.mlcllm.generateSync
//
//// Data structures for vector store
//data class ChunkedDocument(val content: String, val metadata: Map<String, String> = emptyMap())
//data class VectorEntry(val embedding: FloatArray, val document: ChunkedDocument)
//
//class RagChatModel(
//    private val context: Context,
//    private val engine: MLCEngine = MLCEngine(),
//    private val modelPath: String = "compiled-model",
//    private val modelLib: String = "libcompiled-model.so"
//) {
//    private val vectorStore: List<VectorEntry>
//
//    init {
//        engine.reload(modelPath, modelLib)
//
//        val kgText = loadKGFromProvider(context)
//        val documents: List<String> = kgText.lines().filter { it.isNotBlank() }
//        val splitDocs = splitDocuments(documents)
//        vectorStore = buildVectorStore(splitDocs)
//
//        Log.d("RAG", "Vector store created with ${vectorStore.size} entries.")
//    }
//
//    private fun loadKGFromProvider(context: Context): String {
//        val uri = Uri.parse("content://com.example.knowledgegraph.kgprovider/knowledge_graph")
//        val cursor = context.contentResolver.query(uri, null, null, null, null)
//
//        val builder = StringBuilder()
//        cursor?.use {
//            val subjectIndex = it.getColumnIndexOrThrow("subject")
//            val predicateIndex = it.getColumnIndexOrThrow("predicate")
//            val objectIndex = it.getColumnIndexOrThrow("object")
//
//            while (it.moveToNext()) {
//                val subject = it.getString(subjectIndex)
//                val predicate = it.getString(predicateIndex)
//                val obj = it.getString(objectIndex)
//                builder.append("$subject,$predicate,$obj\n")
//            }
//        }
//
//        return builder.toString()
//    }
//
//    private fun splitDocuments(
//        rawDocuments: List<String>,
//        chunkSize: Int = 1000,
//        chunkOverlap: Int = 100
//    ): List<ChunkedDocument> {
//        return rawDocuments.flatMap { doc ->
//            val lines = doc.lines().filter { it.isNotBlank() }
//            val chunks = mutableListOf<ChunkedDocument>()
//            var start = 0
//            while (start < lines.size) {
//                val end = (start + chunkSize).coerceAtMost(lines.size)
//                val chunk = lines.subList(start, end).joinToString("\n")
//                chunks.add(ChunkedDocument(content = chunk))
//                start += (chunkSize - chunkOverlap)
//            }
//            chunks
//        }
//    }
//
//    private fun buildVectorStore(documents: List<ChunkedDocument>): List<VectorEntry> {
//        return documents.mapNotNull { doc ->
//            val embedding = engine.getEmbedding(doc.content)
//            if (embedding != null) VectorEntry(embedding, doc) else null
//        }
//    }
//
//    private fun cosineSimilarity(vec1: FloatArray, vec2: FloatArray): Float {
//        val dot = vec1.zip(vec2).fold(0f) { acc, (a, b) -> acc + (a * b) }
//        val normA = sqrt(vec1.fold(0f) { acc, x -> acc + x * x })
//        val normB = sqrt(vec2.fold(0f) { acc, x -> acc + x * x })
//        return dot / (normA * normB + 1e-8f)
//    }
//
//    private fun retrieveTopK(query: String, k: Int = 5): List<ChunkedDocument> {
//        val queryEmbedding = engine.getEmbedding(query) ?: return emptyList()
//        return vectorStore
//            .map { entry ->
//                val score = cosineSimilarity(queryEmbedding, entry.embedding)
//                Pair(score, entry.document)
//            }
//            .sortedByDescending { it.first }
//            .take(k)
//            .map { it.second }
//    }
//
//    fun runRAGQuery(query: String): String {
//        val topDocs = retrieveTopK(query, k = 5)
//        val contextText = topDocs.joinToString("\n") { it.content }
//
//        val prompt = """
//            Retrieve the answer from the knowledge graph context below and generate a concise response to the query.
//
//            Context:
//            $contextText
//
//            Query:
//            $query
//        """.trimIndent()
//
//        return engine.generateSync(prompt)
//    }
//}