#include <gtest/gtest.h>

#include <brezel/tensor/memory_manager.hpp>
#include <future>
#include <thread>
#include <vector>

using namespace brezel::tensor::memory;

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MemoryManagerTest, DefaultAllocator) {
    DefaultAllocator allocator;

    void* ptr = allocator.allocate(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(allocator.get_allocated_bytes(), 1024);

    std::memset(ptr, 0x42, 1024);
    uint8_t* byte_ptr = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(byte_ptr[i], 0x42);
    }

    allocator.deallocate(ptr, 1024);
    EXPECT_EQ(allocator.get_allocated_bytes(), 0);
}

TEST_F(MemoryManagerTest, AlignedAllocator) {
    DefaultAllocator allocator;

    void* ptr = allocator.allocate(1024, Alignment::AVX);
    EXPECT_NE(ptr, nullptr);

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % static_cast<size_t>(Alignment::AVX), 0);

    allocator.deallocate(ptr, 1024);
}

TEST_F(MemoryManagerTest, ZeroMemoryFlag) {
    DefaultAllocator allocator;

    void* ptr =
        allocator.allocate(1024, Alignment::Default, AllocFlags::ZeroMemory);
    EXPECT_NE(ptr, nullptr);

    uint8_t* byte_ptr = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(byte_ptr[i], 0);
    }

    allocator.deallocate(ptr, 1024);
}

TEST_F(MemoryManagerTest, PoolAllocator) {
    const size_t block_size = 64;
    const size_t initial_capacity = 10;

    PoolAllocator pool(block_size, initial_capacity);
    void* ptr1 = pool.allocate(block_size);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(pool.get_allocated_bytes(), block_size);

    std::memset(ptr1, 0x42, block_size);

    void* ptr2 = pool.allocate(block_size);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr2, ptr1);
    EXPECT_EQ(pool.get_allocated_bytes(), 2 * block_size);

    pool.deallocate(ptr1);
    EXPECT_EQ(pool.get_allocated_bytes(), block_size);

    void* ptr3 = pool.allocate(block_size);
    EXPECT_EQ(ptr3, ptr1);
    EXPECT_EQ(pool.get_allocated_bytes(), 2 * block_size);

    EXPECT_THROW(pool.allocate(block_size + 1),
                 brezel::core::error::RuntimeError);
 
    pool.deallocate(ptr2);
    pool.deallocate(ptr3);
    EXPECT_EQ(pool.get_allocated_bytes(), 0);
}

TEST_F(MemoryManagerTest, PoolAllocatorGrowth) {
    const size_t block_size = 32;
    const size_t initial_capacity = 2;

    PoolAllocator pool(block_size, initial_capacity);

    void* ptr1 = pool.allocate(block_size);
    void* ptr2 = pool.allocate(block_size);
    void* ptr3 = pool.allocate(block_size);
    EXPECT_NE(ptr3, nullptr);

    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
    pool.deallocate(ptr3);
}

TEST_F(MemoryManagerTest, AllocatedPtr) {
    auto int_ptr =
        make_allocated<int>(MemoryManager::instance().default_allocator());
    EXPECT_NE(int_ptr.get(), nullptr);

    *int_ptr = 42;
    EXPECT_EQ(*int_ptr, 42);
    EXPECT_TRUE(int_ptr.allocator() != nullptr);

    int_ptr.reset();
    EXPECT_EQ(int_ptr.get(), nullptr);
}

TEST_F(MemoryManagerTest, AllocatedPtrWithArgs) {
    auto str_ptr = make_allocated<std::string>(
        MemoryManager::instance().default_allocator(), Alignment::Default,
        "Hello, World!");

    EXPECT_EQ(*str_ptr, "Hello, World!");
}

TEST_F(MemoryManagerTest, AllocatedArray) {
    constexpr size_t array_size = 10;

    auto arr_ptr = make_allocated_array<float>(
        array_size, MemoryManager::instance().default_allocator());

    EXPECT_NE(arr_ptr.get(), nullptr);
    EXPECT_EQ(arr_ptr.size(), array_size * sizeof(float));

    for (size_t i = 0; i < array_size; i++) {
        arr_ptr.get()[i] = static_cast<float>(i);
    }

    for (size_t i = 0; i < array_size; i++) {
        EXPECT_EQ(arr_ptr.get()[i], static_cast<float>(i));
    }
}

TEST_F(MemoryManagerTest, AllocatedArrayWithValue) {
    constexpr size_t array_size = 10;

    auto arr_ptr = make_allocated_array<int>(
        array_size, MemoryManager::instance().default_allocator(),
        Alignment::Default, 42);

    for (size_t i = 0; i < array_size; i++) {
        EXPECT_EQ(arr_ptr.get()[i], 42);
    }
}