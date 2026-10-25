#include <gtest/gtest.h>

#include "json_ffi/image_utils.h"
#include "support/utils.h"

TEST(SupportUtils, Base64MetadataWireFormat) {
  EXPECT_EQ(mlc::llm::Base64Encode("abc"), "AwAAAAAAAABhYmM=");
  EXPECT_EQ(mlc::llm::Base64Decode("AwAAAAAAAABhYmM="), "abc");
  for (const auto& value : {std::string(), std::string("a"), std::string("ab"),
                            std::string("a\0b", 3), std::string("[1,2,3]")}) {
    EXPECT_EQ(mlc::llm::Base64Decode(mlc::llm::Base64Encode(value)), value);
  }
}

TEST(SupportUtils, Base64Image) {
  auto image = mlc::llm::json_ffi::LoadImageFromBase64(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
      "x8AAwMCAO+aP1sAAAAASUVORK5CYII=");
  ASSERT_TRUE(image.IsOk());
  auto tensor = image.Unwrap();
  EXPECT_EQ(tensor.shape()[0], 1);
  EXPECT_EQ(tensor.shape()[1], 1);
  EXPECT_EQ(tensor.shape()[2], 3);
}
