#include "detected-object.hpp"

namespace ano
{
    DetectedObjectsVector::iterator DetectedObjectsVectorGetByPixelID(DetectedObjectsVector &detected_objects, unsigned char id_pixel)
    {
        return std::find_if(detected_objects.begin(), detected_objects.end(), [&id_pixel](auto &object) -> bool
                            { return object.id_pixel == id_pixel; });
    }

    void DetectedObjectsVectorSetClass(DetectedObjectsVector &detected_objects, unsigned char id_pixel, unsigned char id_class)
    {
        auto object_it = DetectedObjectsVectorGetByPixelID(detected_objects, id_pixel);

        if (object_it != detected_objects.end())
        {
            object_it->id_class = id_class;
        }
    }
}
