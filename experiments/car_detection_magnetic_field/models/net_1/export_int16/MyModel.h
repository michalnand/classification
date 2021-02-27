#ifndef _MyModel_H_
#define _MyModel_H_


#include <ModelInterface.h>

class MyModel : public ModelInterface<int16_t>
{
	public:
		 MyModel();
		 void forward();
};


#endif
