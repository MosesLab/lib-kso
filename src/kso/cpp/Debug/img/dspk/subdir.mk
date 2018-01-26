################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../img/dspk/dspk.cpp 

O_SRCS += \
../img/dspk/dspk.o 

OBJS += \
./img/dspk/dspk.o 

CPP_DEPS += \
./img/dspk/dspk.d 


# Each subdirectory must supply rules for building sources it contributes
img/dspk/%.o: ../img/dspk/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/python3.5m -I../ -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


