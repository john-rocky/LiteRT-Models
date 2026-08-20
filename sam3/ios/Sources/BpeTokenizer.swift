import Foundation

/// CLIP byte-level BPE tokenizer (vocab.json + merges.txt from facebook/sam3.1),
/// matching sam3's open_clip SimpleTokenizer for typical prompts: lowercase,
/// contraction split ('s 't 're 've 'm 'll 'd), letter / digit / punctuation runs,
/// byte-to-unicode mapping, BPE merges, "</w>" word endings. SAM3 convention:
/// context 32, ZERO padding (id 0), mask = (id == 0). Same reference vectors as
/// the Android port: "wheel" -> [49406, 6744, 49407],
/// "paper bag" -> [49406, 2802, 3365, 49407].
final class BpeTokenizer {
    static let bos = 49406
    static let eot = 49407
    static let maxLen = 32

    private var encoder: [String: Int] = [:]
    private var ranks: [String: Int] = [:]  // "left right" -> merge rank
    private var byteToUnicode: [Character] = []
    private let pieceRegex = try! NSRegularExpression(
        pattern: #"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#)

    init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        guard let vocab = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int]
        else { throw LiteRTError.interface("vocab.json is not a {token: id} object") }
        encoder = vocab

        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
        var rank = 0
        for line in merges.split(separator: "\n").dropFirst() {
            let p = line.split(separator: " ")
            if p.count == 2 { ranks["\(p[0]) \(p[1])"] = rank }
            rank += 1
        }

        // GPT-2 byte-to-unicode table
        var bs: [Int] = []
        bs.append(contentsOf: Int(UnicodeScalar("!").value)...Int(UnicodeScalar("~").value))
        bs.append(contentsOf: 0xA1...0xAC)
        bs.append(contentsOf: 0xAE...0xFF)
        var cs = bs
        var n = 0
        for b in 0...255 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(UnicodeScalar(c)!) }
        byteToUnicode = table
    }

    /// BPE over one mapped word that already carries its "</w>" suffix.
    private func bpe(_ token: String) -> [String] {
        guard token.count > 4 else { return [token] }
        var word = token.dropLast(4).map { String($0) }
        if word.isEmpty { return [token] }
        word[word.count - 1] += "</w>"
        while word.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(word.count - 1) {
                if let r = ranks["\(word[i]) \(word[i + 1])"], r < bestRank {
                    bestRank = r
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }
            let pair = (word[bestIndex], word[bestIndex + 1])
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                    merged.append(pair.0 + pair.1)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }
        return word
    }

    /// prompt -> 32 ids: [BOS, tokens..., EOT, 0-pad...]. Pad mask = (id == 0).
    func encode(_ text: String) -> [Int] {
        // iOS smart punctuation types U+2019 for apostrophes; the CLIP vocab is ASCII.
        let clean = text.lowercased().replacingOccurrences(of: "\u{2019}", with: "'")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        var ids: [Int] = []
        let ns = clean as NSString
        let matches = pieceRegex.matches(in: clean, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            let piece = ns.substring(with: m.range)
            let mapped = String(piece.utf8.map { byteToUnicode[Int($0)] }) + "</w>"
            for t in bpe(mapped) {
                if let id = encoder[t] {
                    ids.append(id)
                } else {
                    for ch in t { if let id = encoder[String(ch)] { ids.append(id) } }
                }
            }
        }
        var out = [Int](repeating: 0, count: Self.maxLen)  // zero padded
        out[0] = Self.bos
        let n = min(ids.count, Self.maxLen - 2)
        for i in 0..<n { out[i + 1] = ids[i] }
        out[n + 1] = Self.eot
        return out
    }
}
