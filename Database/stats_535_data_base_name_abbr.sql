-- MySQL dump 10.13  Distrib 8.0.19, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: stats_535_data_base
-- ------------------------------------------------------
-- Server version	8.0.19

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `name_abbr`
--

DROP TABLE IF EXISTS `name_abbr`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `name_abbr` (
  `STATE` text,
  `abbr` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `name_abbr`
--

LOCK TABLES `name_abbr` WRITE;
/*!40000 ALTER TABLE `name_abbr` DISABLE KEYS */;
INSERT INTO `name_abbr` VALUES ('Alabama','AL'),('Alaska','AK'),('American Samoa','AS'),('Arizona','AZ'),('Arkansas','AR'),('California','CA'),('Colorado','CO'),('Connecticut','CT'),('Delaware','DE'),('District Of Columbia','DC'),('Federated States Of Micronesia','FM'),('Florida','FL'),('Georgia','GA'),('Guam','GU'),('Hawaii','HI'),('Idaho','ID'),('Illinois','IL'),('Indiana','IN'),('Iowa','IA'),('Kansas','KS'),('Kentucky','KY'),('Louisiana','LA'),('Maine','ME'),('Marshall Islands','MH'),('Maryland','MD'),('Massachusetts','MA'),('Michigan','MI'),('Minnesota','MN'),('Mississippi','MS'),('Missouri','MO'),('Montana','MT'),('Nebraska','NE'),('Nevada','NV'),('New Hampshire','NH'),('New Jersey','NJ'),('New Mexico','NM'),('New York','NY'),('North Carolina','NC'),('North Dakota','ND'),('Northern Mariana Islands','MP'),('Ohio','OH'),('Oklahoma','OK'),('Oregon','OR'),('Palau','PW'),('Pennsylvania','PA'),('Puerto Rico','PR'),('Rhode Island','RI'),('South Carolina','SC'),('South Dakota','SD'),('Tennessee','TN'),('Texas','TX'),('Utah','UT'),('Vermont','VT'),('Virgin Islands','VI'),('Virginia','VA'),('Washington','WA'),('West Virginia','WV'),('Wisconsin','WI'),('Wyoming','WY');
/*!40000 ALTER TABLE `name_abbr` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-04-09 16:55:12
