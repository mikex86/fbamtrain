#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <locale>
#include <stdexcept>

using json = nlohmann::json;

fbamtrain::TokenizerHandle::TokenizerHandle(const Tokenizer &tokenizer)
    : tokenizer_(&tokenizer), current_node_(tokenizer.root_.get())
{
}

void fbamtrain::TokenizerHandle::addChar(const char32_t ch)
{
    if (current_node_ == nullptr)
    {
        return; // already in dead state
    }

    const auto child_it = current_node_->children.find(ch);
    if (child_it == current_node_->children.end())
    {
        current_node_ = nullptr; // dead state
        return;
    }
    current_node_ = child_it->second.get();
}

std::optional<size_t> fbamtrain::TokenizerHandle::currentTokenId() const
{
    if (current_node_ == nullptr || !current_node_->token_id.has_value())
    {
        return std::nullopt;
    }
    return current_node_->token_id.value();
}

bool fbamtrain::TokenizerHandle::isDead() const { return current_node_ == nullptr; }

bool fbamtrain::TokenizerHandle::hasContinuation() const
{
    return current_node_ != nullptr && !current_node_->children.empty();
}

void fbamtrain::TokenizerHandle::reset()
{
    if (tokenizer_ == nullptr)
    {
        current_node_ = nullptr;
        return;
    }
    current_node_ = tokenizer_->root_.get();
}

fbamtrain::Tokenizer::Tokenizer(std::vector<std::u32string> tokens)
    : tokens_(std::move(tokens)), root_(std::make_unique<TokenizerNode>())
{
    for (size_t idx = 0; idx < tokens_.size(); ++idx)
    {
        auto *node = root_.get();
        for (const auto ch : tokens_[idx])
        {
            auto &child = node->children[ch];
            if (!child)
            {
                child = std::make_unique<TokenizerNode>();
            }
            node = child.get();
        }
        node->token_id = idx;
    }
}

fbamtrain::Tokenizer fbamtrain::Tokenizer::FromFile(const std::string &path)
{
    std::ifstream file_stream(path);
    if (!file_stream.is_open())
    {
        throw std::runtime_error("Failed to open tokenizer file: " + path);
    }

    std::string header_line{};
    if (!std::getline(file_stream, header_line))
    {
        throw std::runtime_error("Tokenizer file is empty: " + path);
    }
    if (header_line != "# FBAM_TOK v1.0")
    {
        throw std::runtime_error("Invalid tokenizer header in file " + path + ": " + header_line);
    }

    std::vector<std::u32string> tokens{};
    std::string line{};
    size_t line_no = 2;
    while (std::getline(file_stream, line))
    {
        if (line.empty())
        {
            line_no++;
            continue;
        }
        try
        {
            const auto token_utf8 = json::parse(line).get<std::string>();
            tokens.push_back(Utf8ToU32(token_utf8));
        }
        catch (const std::exception &ex)
        {
            throw std::runtime_error("Failed to parse token on line " + std::to_string(line_no) + " in " + path +
                                     ": " + ex.what());
        }
        line_no++;
    }

    std::stable_sort(tokens.begin(), tokens.end(),
                     [](const std::u32string &a, const std::u32string &b) { return a.size() > b.size(); });

    return Tokenizer(std::move(tokens));
}

fbamtrain::TokenizerHandle fbamtrain::Tokenizer::newHandle() const { return TokenizerHandle(*this); }

const std::vector<std::u32string> &fbamtrain::Tokenizer::tokens() const { return tokens_; }

std::u32string fbamtrain::Tokenizer::Utf8ToU32(const std::string &utf8_str)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}
