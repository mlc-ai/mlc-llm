package ai.mlc.mlcchat

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlin.math.sqrt
import ai.mlc.mlcllm.MLCEngine
import ai.mlc.mlcllm.generateSync
import ai.mlc.mlcllm.getEmbedding
import java.io.File
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.*

data class EmbeddedText(val embedding: FloatArray, val text: String)

class RagChatModel(private val context: Context) {

    private var embeddingList: List<EmbeddedText> = emptyList()
    private val prefs = context.getSharedPreferences("RAG_PREFS", Context.MODE_PRIVATE)

    fun loadEmbeddingsIfNeeded() {
        val file = File(context.getExternalFilesDir(null), "Knowledge_graph.vec")
        if (!file.exists()) return

        val currentTimestamp = file.lastModified()
        val savedTimestamp = prefs.getLong("last_vec_timestamp", 0)

        if (currentTimestamp != savedTimestamp) {
            embeddingList = loadVecFromKGProvider(context)
            prefs.edit().putLong("last_vec_timestamp", currentTimestamp).apply()
            Log.d("RAG", "Embeddings reloaded (file updated).")
        } else {
            Log.d("RAG", "Using cached embeddings (no update).")
        }
    }

    fun clearEmbeddings() {
        embeddingList = emptyList()
        prefs.edit().remove("last_vec_timestamp").apply()
        Log.d("RAG", "Embedding cache cleared.")
    }

    fun runRAGQuery(query: String, engine: MLCEngine): String {
        Log.d("RAG_DEBUG", "runRAGQuery called with query: $query")

        if (embeddingList.isEmpty()) {
            embeddingList = loadVecFromKGProvider(context)
        }

        Log.d("RAG_DEBUG", "Embedding list size after load: ${embeddingList.size}")

        val topChunks = retrieveTopK(query, engine, embeddingList, k = 5)
        Log.d("RAG_DEBUG", "Top chunks retrieved: ${topChunks.map { it.text }}")

        val formattedEvents = topChunks.joinToString("\n") {
            "- ${formatEvent(it.text)}"
        }

        val prompt = """
            You are a helpful assistant.
            
            Below is a list of structured calendar facts, including events, holidays, meetings, reminders, and personal activities.
            
            Analyze these entries and answer the user's question as clearly and specifically as possible.
            Only include directly relevant results with specific dates, times, or locations if available.
            
            Calendar Data:
            ${topChunks.joinToString("\n") { "- ${it.text}" }}
            
            User Question:
            $query
            
            Answer:
            """.trimIndent()

        Log.d("RAG_DEBUG", "Final prompt to LLM:\n$prompt")

        return engine.generateSync(prompt)
    }

    private fun formatEvent(raw: String): String {
        return try {
            if (!raw.contains("starts at")) {
                // No timestamp in this entry, return as-is
                return raw
            }

            val inputFormatter = DateTimeFormatter.ofPattern("EEE MMM dd HH:mm:ss z yyyy", Locale.US)
            val outputFormatter = DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy 'at' hh:mm a z", Locale.US)

            val dateStr = raw.substringAfter("starts at ").trim()
            val parsed = ZonedDateTime.parse(dateStr, inputFormatter)

            val prefix = raw.substringBefore("starts at").trim()
            "$prefix starts at ${parsed.format(outputFormatter)}"
        } catch (e: Exception) {
            Log.e("RAG_FORMAT", "Failed to parse event: $raw", e)
            raw
        }
    }

    private fun loadVecFromKGProvider(context: Context): List<EmbeddedText> {
        val uri = Uri.parse("content://com.example.knowledgegraph.kgprovider/knowledge_graph_vec")
        val list = mutableListOf<EmbeddedText>()

        try {
            context.contentResolver.openInputStream(uri)?.bufferedReader()?.useLines { lines ->
                lines.forEachIndexed { index, line ->
                    val parts = line.split("\t")
                    if (parts.size != 2) return@forEachIndexed

                    val embeddingStr = parts[0]
                    val text = parts[1]

                    try {
                        val embeddingList = embeddingStr.split(",")
                            .mapNotNull { it.toFloatOrNull() }

                        if (embeddingList.isNotEmpty()) {
                            val norm = sqrt(embeddingList.fold(0f) { acc, x -> acc + x * x })
                            val normalizedEmbedding = if (norm != 0f)
                                embeddingList.map { it / norm }.toFloatArray()
                            else
                                embeddingList.toFloatArray()

                            list.add(EmbeddedText(normalizedEmbedding, text))
                        }

                    } catch (e: Exception) {
                        Log.e("RAG_PARSE", "Failed to parse embedding at line $index: ${e.message}")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("RAG_FILE", "Failed to open .vec file: ${e.message}")
        }

        return list
    }

    private fun cosineSimilarity(vec1: FloatArray, vec2: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in vec1.indices) {
            dot += vec1[i] * vec2[i]
            normA += vec1[i] * vec1[i]
            normB += vec2[i] * vec2[i]
        }
        return dot / (sqrt(normA) * sqrt(normB) + 1e-8f)
    }

    private fun retrieveTopK(
        query: String,
        engine: MLCEngine,
        data: List<EmbeddedText>,
        k: Int
    ): List<EmbeddedText> {
        val normalizedQuery = normalizeText(query)
        val queryVec = engine.getEmbedding(normalizedQuery) ?: return emptyList()

        fun phraseOverlapScore(query: String, text: String): Float {
            val queryTokens = query.split(" ").filter { it.length > 2 }.toSet()
            val textTokens = normalizeText(text).split(" ").toSet()
            val common = queryTokens.intersect(textTokens)
            return common.size.toFloat() / (queryTokens.size + 1e-5f)
        }

        val scoredData = data.map {
            val cosine = cosineSimilarity(queryVec, it.embedding)
            val overlap = phraseOverlapScore(normalizedQuery, it.text)
            val finalScore = 0.7f * cosine + 0.3f * overlap
            it to finalScore
        }

        val forcedMatches = data.filter {
            normalizedQuery.split(" ").any { word ->
                word.length > 3 && it.text.lowercase().contains(word)
            }
        }

        val topScored = scoredData
            .filterNot { forcedMatches.contains(it.first) }
            .sortedByDescending { it.second }
            .map { it.first }
            .take((k - forcedMatches.size).coerceAtLeast(0))

        val finalContext = (forcedMatches + topScored).distinctBy { it.text }

        finalContext.forEach {
            Log.d("RAG_SIMILARITY", "Included in final context: '${it.text}'")
        }

        return finalContext
    }

    fun normalizeText(text: String): String {
        return text.lowercase()
            .replace("-", " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }
}