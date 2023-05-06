
# -----------------------------------------------------------------------------------------
add_executable(18_RNN)
target_sources(18_RNN PRIVATE RNN/main.cpp
                              RNN/rnn.cpp
                              RNN/rnn.h
)

target_link_libraries(18_RNN ${TORCH_LIBRARIES})
set_target_properties(18_RNN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(18_bi_RNN)
target_sources(18_bi_RNN PRIVATE bi_RNN/main.cpp
                                 bi_RNN/bi_rnn.cpp
                                 bi_RNN/bi_rnn.h
)

target_link_libraries(18_bi_RNN ${TORCH_LIBRARIES})
set_target_properties(18_bi_RNN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(18_RNN_LM)
target_sources(18_RNN_LM PRIVATE RNN_LM/main.cpp
                                 RNN_LM/rnn_lm.cpp
                                 RNN_LM/corpus.cpp
                                 RNN_LM/dictionary.cpp
                                 RNN_LM/rnn_lm.h
                                 RNN_LM/corpus.h
                                 RNN_LM/dictionary.h
)

target_link_libraries(18_RNN_LM ${TORCH_LIBRARIES})
set_target_properties(18_RNN_LM PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
