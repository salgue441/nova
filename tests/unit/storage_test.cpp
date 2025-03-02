#include <gtest/gtest.h>

#include <brezel/core/error/error.hpp>
#include <brezel/tensor/storage.hpp>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

using namespace brezel::tensor;
using namespace brezel::tensor::memory;
using namespace brezel::core::error;

class StorageTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(StorageTest, DefaultConstructorCreatesEmptyStorage) {
    Storage<float> storage;

    EXPECT_EQ(storage.size(), 0);
    EXPECT_EQ(storage.element_size(), sizeof(float));
    EXPECT_EQ(storage.nbytes(), 0);
    EXPECT_EQ(storage.device(), DeviceType::CPU);
    EXPECT_TRUE(storage.empty());
    EXPECT_TRUE(storage.is_contiguous());
    EXPECT_EQ(storage.data(), nullptr);
}

TEST_F(StorageTest, ConstructWithSize) {
    const size_t size = 100;
    Storage<int> storage(size);

    EXPECT_EQ(storage.size(), size);
    EXPECT_EQ(storage.element_size(), sizeof(int));
    EXPECT_EQ(storage.nbytes(), size * sizeof(int));
    EXPECT_FALSE(storage.empty());
    EXPECT_TRUE(storage.is_contiguous());
    EXPECT_NE(storage.data(), nullptr);
}

TEST_F(StorageTest, ConstructWithSizeAndValue) {
    const size_t size = 100;
    const int value = 42;
    Storage<int> storage(size, value);

    EXPECT_EQ(storage.size(), size);
    EXPECT_FALSE(storage.empty());

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(storage[i], value);
    }
}

TEST_F(StorageTest, CopyConstruction) {
    const size_t size = 100;
    Storage<double> original(size);

    for (size_t i = 0; i < size; ++i) {
        original[i] = static_cast<double>(i);
    }

    Storage<double> copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.device(), original.device());
    EXPECT_NE(copy.data(), original.data());  // Different memory address

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(copy[i], original[i]);
    }

    copy[0] = 999.0;
    EXPECT_EQ(copy[0], 999.0);
    EXPECT_EQ(original[0], 0.0);
}

TEST_F(StorageTest, MoveConstruction) {
    const size_t size = 100;
    Storage<int> original(size, 42);
    void* original_data = original.data();

    Storage<int> moved(std::move(original));
    EXPECT_EQ(original.size(), 0);
    EXPECT_TRUE(original.empty());
    EXPECT_EQ(original.data(), nullptr);

    EXPECT_EQ(moved.size(), size);
    EXPECT_FALSE(moved.empty());
    EXPECT_EQ(moved.data(), original_data);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(moved[i], 42);
    }
}

TEST_F(StorageTest, CopyAssignment) {
    const size_t size = 50;
    Storage<float> original(size, 3.14f);
    Storage<float> copy;

    copy = original;

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_NE(copy.data(), original.data());

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(copy[i], 3.14f);
    }
}

TEST_F(StorageTest, MoveAssignment) {
    const size_t size = 50;
    Storage<float> original(size, 3.14f);
    void* original_data = original.data();
    Storage<float> target;

    target = std::move(original);

    EXPECT_EQ(target.size(), size);
    EXPECT_EQ(target.data(), original_data);
    EXPECT_EQ(original.size(), 0);
    EXPECT_EQ(original.data(), nullptr);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(target[i], 3.14f);
    }
}

TEST_F(StorageTest, ElementAccess) {
    const size_t size = 100;
    Storage<int> storage(size);

    for (size_t i = 0; i < size; ++i) {
        storage[i] = static_cast<int>(i);
    }

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(storage[i], i);
        EXPECT_EQ(storage.at(i), i);
    }

    EXPECT_THROW(storage.at(size), LogicError);
}

TEST_F(StorageTest, Resize) {
    const size_t initial_size = 10;
    Storage<int> storage(initial_size, 42);

    const size_t larger_size = 20;
    storage.resize(larger_size);

    EXPECT_EQ(storage.size(), larger_size);
    for (size_t i = 0; i < initial_size; ++i) {
        EXPECT_EQ(storage[i], 42);
    }

    const size_t smaller_size = 5;
    storage.resize(smaller_size);
    EXPECT_EQ(storage.size(), smaller_size);

    for (size_t i = 0; i < smaller_size; ++i) {
        EXPECT_EQ(storage[i], 42);
    }

    storage.resize(0);
    EXPECT_EQ(storage.size(), 0);
    EXPECT_TRUE(storage.empty());

    storage.resize(5);
    EXPECT_EQ(storage.size(), 5);
    EXPECT_FALSE(storage.empty());
}

TEST_F(StorageTest, FillMethod) {
    const size_t size = 10;
    Storage<double> storage(size);

    storage.fill(3.14);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(storage[i], 3.14);
    }
}

TEST_F(StorageTest, CloneMethod) {
    const size_t size = 10;
    Storage<int> storage(size, 42);

    std::unique_ptr<StorageBase> cloned = storage.clone();
    EXPECT_EQ(cloned->size(), size);
    EXPECT_EQ(cloned->element_size(), sizeof(int));

    Storage<int>* typed_clone = dynamic_cast<Storage<int>*>(cloned.get());
    ASSERT_NE(typed_clone, nullptr);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ((*typed_clone)[i], 42);
    }
}

TEST_F(StorageTest, CreateFromExistingData) {
    const size_t size = 5;
    std::vector<int> data(size);
    std::iota(data.begin(), data.end(), 1);

    Storage<int> storage_copy(data.data(), size, true);
    EXPECT_EQ(storage_copy.size(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(storage_copy[i], i + 1);
    }

    data[0] = 99;
    EXPECT_EQ(storage_copy[0], 1);
}

TEST_F(StorageTest, DeviceTypeTest) {
    Storage<float> storage(10);
    EXPECT_EQ(storage.device(), DeviceType::CPU);

    Storage<float> gpu_storage(
        10, MemoryManager::instance().default_allocator(), DeviceType::CUDA);
    EXPECT_EQ(gpu_storage.device(), DeviceType::CUDA);
}

// SharedStorage tests
TEST_F(StorageTest, SharedStorageDefaultConstructor) {
    SharedStorage shared;

    EXPECT_EQ(shared.size(), 0);
    EXPECT_EQ(shared.nbytes(), 0);
    EXPECT_EQ(shared.data(), nullptr);
    EXPECT_TRUE(shared.empty());
    EXPECT_FALSE(static_cast<bool>(shared));
    EXPECT_EQ(shared.use_count(), 0);
}

TEST_F(StorageTest, SharedStorageFromUniquePtr) {
    auto storage = std::make_unique<Storage<int>>(10, 42);
    SharedStorage shared(std::move(storage));

    EXPECT_EQ(shared.size(), 10);
    EXPECT_EQ(shared.element_size(), sizeof(int));
    EXPECT_TRUE(static_cast<bool>(shared));
    EXPECT_EQ(shared.use_count(), 1);
    EXPECT_TRUE(shared.unique());
}

TEST_F(StorageTest, SharedStorageCopy) {
    auto storage = std::make_unique<Storage<int>>(10, 42);
    SharedStorage shared1(std::move(storage));
    SharedStorage shared2(shared1);

    EXPECT_EQ(shared1.get(), shared2.get());
    EXPECT_EQ(shared1.use_count(), 2);
    EXPECT_EQ(shared2.use_count(), 2);
    EXPECT_FALSE(shared1.unique());
    EXPECT_FALSE(shared2.unique());
}

TEST_F(StorageTest, SharedStorageMove) {
    auto storage = std::make_unique<Storage<int>>(10, 42);
    StorageBase* storage_ptr = storage.get();
    SharedStorage shared1(std::move(storage));
    SharedStorage shared2(std::move(shared1));

    // Shared2 should now own the storage
    EXPECT_EQ(shared2.get(), storage_ptr);
    EXPECT_EQ(shared2.use_count(), 1);
    EXPECT_TRUE(shared2.unique());

    // Shared1 should be empty
    EXPECT_EQ(shared1.get(), nullptr);
    EXPECT_EQ(shared1.use_count(), 0);
    EXPECT_TRUE(shared1.empty());
}

TEST_F(StorageTest, SharedStorageCopyAssignment) {
    auto storage1 = std::make_unique<Storage<int>>(10, 1);
    auto storage2 = std::make_unique<Storage<int>>(20, 2);
    SharedStorage shared1(std::move(storage1));
    SharedStorage shared2(std::move(storage2));

    shared2 = shared1;
    EXPECT_EQ(shared1.get(), shared2.get());
    EXPECT_EQ(shared1.size(), 10);
    EXPECT_EQ(shared2.size(), 10);
    EXPECT_EQ(shared1.use_count(), 2);
    EXPECT_EQ(shared2.use_count(), 2);
}

TEST_F(StorageTest, SharedStorageMoveAssignment) {
    auto storage1 = std::make_unique<Storage<int>>(10, 1);
    auto storage2 = std::make_unique<Storage<int>>(20, 2);
    StorageBase* storage1_ptr = storage1.get();
    SharedStorage shared1(std::move(storage1));
    SharedStorage shared2(std::move(storage2));

    shared2 = std::move(shared1);

    // shared2 should now point to the first storage
    EXPECT_EQ(shared2.get(), storage1_ptr);
    EXPECT_EQ(shared2.size(), 10);
    EXPECT_EQ(shared2.use_count(), 1);

    // shared1 should be empty
    EXPECT_EQ(shared1.get(), nullptr);
    EXPECT_EQ(shared1.use_count(), 0);
}

TEST_F(StorageTest, SharedStorageGetTyped) {
    auto int_storage = std::make_unique<Storage<int>>(10, 42);
    SharedStorage shared(std::move(int_storage));
    Storage<int>* typed_storage = shared.get<int>();

    ASSERT_NE(typed_storage, nullptr);
    ASSERT_EQ(typed_storage->size(), 10);
    ASSERT_EQ((*typed_storage)[0], 42);
    EXPECT_THROW(shared.get<double>(), LogicError);
}

TEST_F(StorageTest, SharedStorageClone) {
    auto storage = std::make_unique<Storage<int>>(10, 42);
    SharedStorage shared1(std::move(storage));
    SharedStorage shared2 = shared1.clone();

    EXPECT_NE(shared1.get(), shared2.get());
    EXPECT_EQ(shared1.size(), shared2.size());
    EXPECT_EQ(shared1.element_size(), shared2.element_size());

    Storage<int>* typed_storage1 = shared1.get<int>();
    Storage<int>* typed_storage2 = shared2.get<int>();

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ((*typed_storage1)[i], (*typed_storage2)[i]);
    }
}

TEST_F(StorageTest, SharedStorageResize) {
    auto storage = std::make_unique<Storage<int>>(10, 42);
    SharedStorage shared(std::move(storage));

    shared.resize(20);
    EXPECT_EQ(shared.size(), 20);

    Storage<int>* typed_storage = shared.get<int>();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ((*typed_storage)[i], 42);
    }
}

TEST_F(StorageTest, MakeStorageFunctions) {
    auto shared1 = make_storage<int>(10);
    EXPECT_EQ(shared1.size(), 10);
    EXPECT_EQ(shared1.element_size(), sizeof(int));

    // make_storage with size and value
    auto shared2 = make_storage<float>(10, 3.14f);
    EXPECT_EQ(shared2.size(), 10);

    auto typed_storage2 = shared2.get<float>();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ((*typed_storage2)[i], 3.14f);
    }

    // zeros
    auto shared3 = make_zeros_storage<double>(10);
    EXPECT_EQ(shared3.size(), 10);

    auto typed_storage3 = shared3.get<double>();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ((*typed_storage3)[i], 0.0);
    }

    // ones
    auto shared4 = make_ones_storage<int>(10);
    EXPECT_EQ(shared4.size(), 10);

    auto typed_storage4 = shared4.get<int>();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ((*typed_storage4)[i], 1);
    }
}

TEST_F(StorageTest, MultithreadedSharedStorage) {
    const int thread_count = 8;
    const size_t size = 1000;
    auto shared = make_storage<int>(size, 0);
    std::vector<std::thread> threads;

    for (int t = 0; t < thread_count; ++t) {
        threads.emplace_back([&shared, t, size]() {
            auto storage = shared.get<int>();
            size_t start = (size * t) / thread_count;
            size_t end = (size * (t + 1)) / thread_count;

            for (size_t i = start; i < end; ++i) {
                (*storage)[i] += 1;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto storage = shared.get<int>();
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ((*storage)[i], 1);
    }
}