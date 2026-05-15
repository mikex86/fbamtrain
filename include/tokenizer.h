#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fbamtrain
{
    struct TokenizerNode
    {
        std::unordered_map<char32_t, std::unique_ptr<TokenizerNode>> children{};
        std::optional<size_t> token_id{};
    };

    class Tokenizer;

    class TokenizerHandle
    {
      public:
        explicit TokenizerHandle(const Tokenizer &tokenizer);

        void addChar(char32_t ch);

        [[nodiscard]] std::optional<size_t> currentTokenId() const;

        [[nodiscard]] bool isDead() const;

        [[nodiscard]] bool hasContinuation() const;

        void reset();

      private:
        const Tokenizer *tokenizer_{};
        const TokenizerNode *current_node_{};
    };

    class Tokenizer
    {
      public:
        Tokenizer() = default;

        explicit Tokenizer(std::vector<std::u32string> tokens);

        [[nodiscard]] static Tokenizer FromFile(const std::string &path);

        [[nodiscard]] TokenizerHandle newHandle() const;

        [[nodiscard]] const std::vector<std::u32string> &tokens() const;

      private:
        std::vector<std::u32string> tokens_{};
        std::unique_ptr<TokenizerNode> root_{};

        [[nodiscard]] static std::u32string Utf8ToU32(const std::string &utf8_str);

        friend class TokenizerHandle;
    };
} // namespace fbamtrain
