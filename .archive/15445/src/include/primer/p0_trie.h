//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// p0_trie.h
//
// Identification: src/include/primer/p0_trie.h
//
// Copyright (c) 2015-2022, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/exception.h"
#include "common/logger.h"
#include "common/rwlatch.h"

namespace bustub {

class RLock {
  ReaderWriterLatch *latch_;

 public:
  explicit RLock(ReaderWriterLatch *latch) : latch_(latch) { latch_->RLock(); }
  ~RLock() { latch_->RUnlock(); }
};

class WLock {
  ReaderWriterLatch *latch_;

 public:
  explicit WLock(ReaderWriterLatch *latch) : latch_(latch) { latch_->WLock(); }
  ~WLock() { latch_->WUnlock(); }
};

/**
 * TrieNode is a generic container for any node in Trie.
 */
class TrieNode {
 public:
  /**
   * TODO(P0): Add implementation
   *
   * @brief Construct a new Trie Node object with the given key char.
   * is_end_ flag should be initialized to false in this constructor.
   *
   * @param key_char Key character of this trie node
   */

  // NOTE
  // https://www.geeksforgeeks.org/use-of-explicit-keyword-in-cpp
  // Từ khóa explicit dùng để đánh dấu constructor không tự động convert kiểu,
  // khi mà constructor này nhận đầu vào là một tham số. Nếu ko có từ khóa này thì
  // nó sẽ tự động convert kiểu.
  // VD: có thể viết trie_node == 'c' (nếu đã định nghĩa operator ==)

  // Khởi tạo một trie node mới với ký tự key_char, gán flag is_end_ thành false
  // 3 thuộc tính của trie node được định nghĩa ở phía dưới là:
  // * char key_char_;
  // * bool is_end_{false};
  // * std::unordered_map<char, std::unique_ptr<TrieNode>> children_;
  // Vì node này mới được tạo ra nên biến children_ sẽ được mặc định khởi tạo là empty
  explicit TrieNode(char key_char) {
    is_end_ = false;
    key_char_ = key_char;
  }

  /**
   * @brief Move constructor for trie node object. The unique pointers stored
   * in children_ should be moved from other_trie_node to new trie node.
   *
   * @param other_trie_node Old trie node.
   */

  // NOTE
  // https://stackoverflow.com/questions/4549151/c-double-address-operator
  // int&& a means "a" is an r-value reference.
  // && is normally only used to declare a parameter of a function.
  // And it only takes a r-value expression. If you don't know what an r-value is,
  // the simple explanation is that it doesn't have a memory address.
  // E.g. the number 6, and character 'v' are both r-values.
  // int a, a is an l-value, however (a+2) is an r-value.
  // https://en.wikipedia.org/wiki/C%2B%2B11#Rvalue_references_and_move_constructors

  // NOTE
  // https://stackoverflow.com/questions/3413470/what-is-stdmove-and-when-should-it-be-used
  // dùng std::move để swap resource thay vì copy
  // Khởi tạo trie node mới từ một node có sẵn
  TrieNode(TrieNode &&other_trie_node) noexcept {
    children_ = std::move(other_trie_node.children_);
    key_char_ = other_trie_node.key_char_;
    is_end_ = other_trie_node.is_end_;
  }

  /**
   * @brief Destroy the TrieNode object.
   */
  virtual ~TrieNode() = default;

  /**
   * @brief Whether this trie node has a child node with specified key char.
   *
   * @param key_char Key char of child node.
   * @return True if this trie node has a child with given key, false otherwise.
   */
  bool HasChild(char key_char) const { return (children_.find(key_char) != children_.end()); }

  /**
   * @brief Whether this trie node has any children at all. This is useful
   * when implementing 'Remove' functionality.
   *
   * @return True if this trie node has any child node, false if it has no child node.
   */
  bool HasChildren() const { return !children_.empty(); }

  /**
   * @brief Whether this trie node is the ending character of a key string.
   *
   * @return True if is_end_ flag is true, false if is_end_ is false.
   */
  bool IsEndNode() const { return is_end_; }

  /**
   * @brief Return key char of this trie node.
   *
   * @return key_char_ of this trie node.
   */
  char GetKeyChar() const { return key_char_; }

  /**
   * @brief Insert a child node for this trie node into children_ map, given the key char and
   * unique_ptr of the child node. If specified key_char already exists in children_,
   * return nullptr. If parameter `child`'s key char is different than parameter
   * `key_char`, return nullptr.
   *
   * Note that parameter `child` is rvalue and should be moved when it is
   * inserted into children_map.
   *
   * The return value is a pointer to unique_ptr because pointer to unique_ptr can access the
   * underlying data without taking ownership of the unique_ptr. Further, we can set the return
   * value to nullptr when error occurs.
   *
   * @param key Key of child node
   * @param child Unique pointer created for the child node. This should be added to children_ map.
   * @return Pointer to unique_ptr of the inserted child node. If insertion fails, return nullptr.
   */
  std::unique_ptr<TrieNode> *InsertChildNode(char key_char, std::unique_ptr<TrieNode> &&child) {
    // https://en.cppreference.com/w/cpp/container/unordered_map/insert
    // When same key is inserted twice, insert should return nullptr
    // When inserted key and unique_ptr's key does not match, return nullptr
    if (key_char != child->GetKeyChar()) {
      return nullptr;
    }
    bool ok = children_.insert({key_char, std::move(child)}).second;
    if (ok) {
      return &children_[key_char];
    }
    return nullptr;
  }

  /**
   * @brief Get the child node given its key char. If child node for given key char does
   * not exist, return nullptr.
   *
   * @param key Key of child node
   * @return Pointer to unique_ptr of the child node, nullptr if child
   *         node does not exist.
   */
  std::unique_ptr<TrieNode> *GetChildNode(char key_char) {
    // https://cplusplus.com/reference/unordered_map/unordered_map/find/#example
    if (children_.empty()) {
      return nullptr;
    }
    if (children_.find(key_char) == children_.end()) {
      return nullptr;
    }
    return &children_[key_char];
  }

  /**
   * @brief Remove child node from children_ map.
   * If key_char does not exist in children_, return immediately.
   *
   * @param key_char Key char of child node to be removed
   */
  void RemoveChildNode(char key_char) {
    // https://cplusplus.com/reference/unordered_map/unordered_map/erase/#example
    children_.erase(children_.find(key_char));
  }

  /**
   * @brief Set the is_end_ flag to true or false.
   *
   * @param is_end Whether this trie node is ending char of a key string
   */
  void SetEndNode(bool is_end) { is_end_ = is_end; }

 protected:
  /** Key character of this trie node */
  char key_char_;
  /** whether this node marks the end of a key */
  bool is_end_{false};
  /** A map of all child nodes of this trie node, which can be accessed by each
   * child node's key char. */
  std::unordered_map<char, std::unique_ptr<TrieNode>> children_;
};

/**
 * TrieNodeWithValue is a node that marks the ending of a key, and it can
 * hold a value of any type T.
 */
template <typename T>
class TrieNodeWithValue : public TrieNode {
 private:
  /* Value held by this trie node. */
  T value_;

 public:
  /**
   * @brief Construct a new TrieNodeWithValue object from a TrieNode object and specify its value.
   * This is used when a non-terminal TrieNode is converted to terminal TrieNodeWithValue.
   *
   * The children_ map of TrieNode should be moved to the new TrieNodeWithValue object.
   * Since it contains unique pointers, the first parameter is a rvalue reference.
   *
   * You should:
   * 1) invoke TrieNode's move constructor to move data from TrieNode to
   * TrieNodeWithValue.
   * 2) set value_ member variable of this node to parameter `value`.
   * 3) set is_end_ to true
   *
   * @param trieNode TrieNode whose data is to be moved to TrieNodeWithValue
   * @param value
   */
  TrieNodeWithValue(TrieNode &&trieNode, T value) : TrieNode(std::forward<TrieNode>(trieNode)) {
    // https://stackoverflow.com/questions/4086800/move-constructor-on-derived-object
    this->value_ = value;
    this->is_end_ = true;
  }

  /**
   * @brief Construct a new TrieNodeWithValue. This is used when a new terminal node is constructed.
   *
   * You should:
   * 1) Invoke the constructor for TrieNode with the given key_char.
   * 2) Set value_ for this node.
   * 3) set is_end_ to true.
   *
   * @param key_char Key char of this node
   * @param value Value of this node
   */
  TrieNodeWithValue(char key_char, T value) : TrieNode(key_char) {
    value_ = value;
    SetEndNode(true);
  }

  /**
   * @brief Destroy the Trie Node With Value object
   */
  ~TrieNodeWithValue() override = default;

  /**
   * @brief Get the stored value_.
   *
   * @return Value of type T stored in this node
   */
  T GetValue() const { return value_; }
};

/**
 * Trie is a concurrent key-value store. Each key is a string and its corresponding
 * value can be any type.
 */
class Trie {
 private:
  /* Root node of the trie */
  std::unique_ptr<TrieNode> root_;
  /* Read-write lock for the trie */
  ReaderWriterLatch latch_;

 public:
  /**
   * @brief Construct a new Trie object. Initialize the root node with '\0'
   * character.
   */
  Trie() {
    root_ = std::make_unique<TrieNode>('\0');
    root_->SetEndNode(false);
  }

  /**
   * @brief Insert key-value pair into the trie.
   *
   * If the key is an empty string, return false immediately.
   *
   * If the key already exists, return false. Duplicated keys are not allowed and
   * you should never overwrite value of an existing key.
   *
   * When you reach the ending character of a key:
   * 1. If TrieNode with this ending character does not exist, create new TrieNodeWithValue
   * and add it to parent node's children_ map.
   * 2. If the terminal node is a TrieNode, then convert it into TrieNodeWithValue by
   * invoking the appropriate constructor.
   * 3. If it is already a TrieNodeWithValue,
   * then insertion fails and returns false. Do not overwrite existing data with new data.
   *
   * You can quickly check whether a TrieNode pointer holds TrieNode or TrieNodeWithValue
   * by checking the is_end_ flag. If is_end_ == false, then it points to TrieNode. If
   * is_end_ == true, it points to TrieNodeWithValue.
   *
   * @param key Key used to traverse the trie and find the correct node
   * @param value Value to be inserted
   * @return True if insertion succeeds, false if the key already exists
   */
  template <typename T>
  bool Insert(const std::string &key, T value) {
    // * If the key is an empty string, return false immediately.
    if (key.empty()) {
      return false;
    }

    WLock lock(&latch_);

    // * If the key already exists, return false. Duplicated keys are not allowed and
    // * you should never overwrite value of an existing key.
    bool success = false;
    GetValue<T>(key, &success);
    if (success) {
      return false;
    }

    auto curr_node = &root_;
    for (char curr_char : key) {
      auto child = (*curr_node)->GetChildNode(curr_char);
      if (child == nullptr) {
        child = (*curr_node)->InsertChildNode(curr_char, std::make_unique<TrieNode>(curr_char));
      }
      curr_node = child;
    }

    // * When you reach the ending character of a key:
    // * 1. If TrieNode with this ending character does not exist, create new TrieNodeWithValue
    // * and add it to parent node's children_ map.
    // * 2. If the terminal node is a TrieNode, then convert it into TrieNodeWithValue by
    // * invoking the appropriate constructor.
    // * 3. If it is already a TrieNodeWithValue,
    // * then insertion fails and returns false. Do not overwrite existing data with new data.
    if ((*curr_node)->IsEndNode()) {
      return false;
    }

    // curr_note is a pointer to std::unique_ptr<TrieNode>
    // => Use `**curr_node` to access to the TrieNode data
    *curr_node = std::make_unique<TrieNodeWithValue<T>>(std::move(**curr_node), value);

    return true;
  }

  /**
   * @brief Remove key value pair from the trie.
   * This function should also remove nodes that are no longer part of another
   * key. If key is empty or not found, return false.
   *
   * You should:
   * 1) Find the terminal node for the given key.
   * 2) If this terminal node does not have any children, remove it from its
   * parent's children_ map.
   * 3) Recursively remove nodes that have no children and are not terminal node
   * of another key.
   *
   * @param key Key used to traverse the trie and find the correct node
   * @return True if the key exists and is removed, false otherwise
   */
  bool Remove(const std::string &key) {
    if (key.empty()) {
      return false;
    }

    WLock lock(&latch_);

    auto curr_node = &root_;
    std::list<std::unique_ptr<TrieNode> *> nodes_path = {};
    for (char curr_char : key) {
      nodes_path.push_front(curr_node);
      curr_node = (*curr_node)->GetChildNode(curr_char);
      if (curr_node == nullptr) {
        LOG_INFO(">>> key `%s` not exists", key.c_str());
        // key ko có trong trie
        return false;
      }
    }

    // Sau khi đi hết char trong key, nếu node cuối là end_node thì key đó có trong trie
    bool remove_node = (*curr_node)->IsEndNode();
    LOG_INFO(">>> key `%s` is not end_node", key.c_str());
    if (!remove_node) {
      return false;
    }

    LOG_INFO(">>> key `%s` removing", key.c_str());

    if ((*curr_node)->HasChildren()) {
      (*curr_node)->SetEndNode(false);
      LOG_DEBUG(">>> has children, set end note to fail");
      return true;
    }

    // Recursively remove nodes that have no children and is not end note
    char remove_char = (*curr_node)->GetKeyChar();
    for (auto node : nodes_path) {
      (*node)->RemoveChildNode(remove_char);
      LOG_DEBUG(">>> remove child `%c`", remove_char);
      if ((*node)->HasChildren() || (*node)->IsEndNode()) {
        break;
      }
      remove_char = (*node)->GetKeyChar();
    }
    return true;
  }

  /**
   * @brief Get the corresponding value of type T given its key.
   * If key is empty, set success to false.
   * If key does not exist in trie, set success to false.
   * If the given type T is not the same as the value type stored in TrieNodeWithValue
   * (ie. GetValue<int> is called but terminal node holds std::string),
   * set success to false.
   *
   * To check whether the two types are the same, dynamic_cast
   * the terminal TrieNode to TrieNodeWithValue<T>. If the casted result
   * is not nullptr, then type T is the correct type.
   *
   * @param key Key used to traverse the trie and find the correct node
   * @param success Whether GetValue is successful or not
   * @return Value of type T if type matches
   */
  template <typename T>
  T GetValue(const std::string &key, bool *success) {
    // * If the key is an empty string, return false immediately.
    if (key.empty()) {
      *success = false;
    }

    // RLock lock(&latch_);

    auto curr_node = &root_;
    for (char curr_char : key) {
      curr_node = (*curr_node)->GetChildNode(curr_char);
      if (curr_node == nullptr) {
        *success = false;
        return {};
      }
    }

    bool is_end_node = (*curr_node)->IsEndNode();
    if (is_end_node) {
      // curr_note is a pointer to std::unique_ptr<TrieNode>
      // => Use `**curr_node` to access to the TrieNode data
      // => Use `&**curr_node` lấy địa chỉ (con trỏ) tới TrieNode data

      // Vì curr_node là end_node nên chắc chắn nó là TrieNodeWithValue
      // Cast con trỏ tới TrieNode data sang TrieNodeWithValue<T> chỉ để xem
      // value có đúng là kiểu T hay không
      auto casted = dynamic_cast<TrieNodeWithValue<T> *>(&**curr_node);

      // Nếu value ko phải kiểu T thì casted sẽ là nullprt
      if (casted == nullptr) {
        *success = false;
        return {};
      }

      // value đúng là kiểu T thì tìm kiếm thành công và trả về giá trị
      *success = true;
      return casted->GetValue();
    }

    // curr_node không phải end_node tức ko tìm thấy key trong trie
    *success = false;
    return {};
  }
};
}  // namespace bustub
