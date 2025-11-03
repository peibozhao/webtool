#pragma once

#include <tuple>
#include <type_traits>

template <typename T> struct GRPCTraits;

template <typename ClassType, typename ReturnType, typename... Args>
struct GRPCTraits<ReturnType (ClassType::*)(Args...)> {
  using Class = ClassType;
  using Return = ReturnType;
  using Request = std::remove_const_t<std::remove_pointer_t<
      std::remove_reference_t<std::tuple_element_t<1, std::tuple<Args...>>>>>;
  using Response = std::remove_pointer_t<
      std::remove_reference_t<std::tuple_element_t<2, std::tuple<Args...>>>>;
};
