
project(Reinforcement_CPP
        LANGUAGES CXX
        VERSION 1.0.0
        DESCRIPTION "Reinforcement learning in C++ using PyTorch"
        )
        
# path to ale include file
set(ALE_INCLUDE_DIRS "/media/hhj/localssd/LibtorchPrgs/arcade-learning-environment/src")

# path to libale.so 
find_library(LIBALE libale.so HINTS "/media/hhj/localssd/LibtorchPrgs/arcade-learning-environment/")

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

add_executable(22_Reinforcement_DQN 
src/main.cpp 
src/ExperienceReplay.cpp 
src/ExperienceReplay.h 
src/dqn.cpp 
src/Trainer.cpp 
src/dqn.h 
src/Trainer.h 
src/PrioritizedExperienceReplay.cpp 
src/PrioritizedExperienceReplay.h 
src/noisy.h 
src/categorical_dqn.h)

target_include_directories(22_Reinforcement_DQN  PRIVATE ${ALE_INCLUDE_DIRS})
target_link_libraries(22_Reinforcement_DQN   ${LIBALE} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(22_Reinforcement_DQN  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#include_directories(${CPPRL_INCLUDE_DIRS})
#find_library(LIBALE libale.so /Users/navneetmadhukumar/Downloads/Arcade-Learning-Environment-master/)
#target_link_libraries(Reinforcement_CPP  ${LIBALE} ${TORCH_LIBRARIES} )
