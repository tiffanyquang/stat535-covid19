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
-- Table structure for table `state_density`
--

DROP TABLE IF EXISTS `state_density`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `state_density` (
  `State` text,
  `Density` double DEFAULT NULL,
  `Pop` int DEFAULT NULL,
  `LandArea` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `state_density`
--

LOCK TABLES `state_density` WRITE;
/*!40000 ALTER TABLE `state_density` DISABLE KEYS */;
INSERT INTO `state_density` VALUES ('District of Columbia',11814.541,720687,61),('New Jersey',1215.1991,8936574,7354),('Rhode Island',1021.4323,1056161,1034),('Massachusetts',894.4355,6976597,7800),('Connecticut',735.8689,3563077,4842),('Maryland',626.6731,6083116,9707),('Delaware',504.3073,982895,1949),('New York',412.5211,19440469,47126),('Florida',410.1256,21992985,53625),('Ohio',287.5038,11747694,40861),('Pennsylvania',286.5449,12820878,44743),('California',256.3727,39937489,155779),('Illinois',228.0243,12659682,55519),('Hawaii',219.9419,1412687,6423),('Virginia',218.4403,8626207,39490),('North Carolina',218.2702,10611862,48618),('Indiana',188.281,6745354,35826),('Georgia',186.6719,10736059,57513),('Michigan',177.6655,10045029,56539),('South Carolina',173.3174,5210095,30061),('Tennessee',167.2748,6897576,41235),('New Hampshire',153.1605,1371246,8953),('Washington',117.3272,7797095,66456),('Kentucky',113.9566,4499692,39486),('Texas',112.8204,29472295,261232),('Wisconsin',108.0497,5851754,54158),('Louisiana',107.5175,4645184,43204),('Alabama',96.9221,4908621,50645),('Missouri',89.7453,6169270,68742),('West Virginia',73.9691,1778070,24038),('Minnesota',71.5922,5700671,79627),('Vermont',68.1416,628061,9217),('Arizona',64.955,7378494,113594),('Mississippi',63.7056,2989260,46923),('Arkansas',58.403,3038999,52035),('Oklahoma',57.6547,3954821,68595),('Iowa',56.9284,3179849,55857),('Colorado',56.4011,5845526,103642),('Oregon',44.8086,4301089,95988),('Maine',43.6336,1345790,30843),('Utah',39.943,3282115,82170),('Kansas',35.5968,2910357,81759),('Nevada',28.5993,3139658,109781),('Nebraska',25.4161,1952570,76824),('Idaho',22.0969,1826156,82643),('New Mexico',17.285,2096640,121298),('South Dakota',11.9116,903027,75811),('North Dakota',11.0393,761723,69001),('Montana',7.4668,1086759,145546),('Wyoming',5.84,567025,97093),('Alaska',1.2863,734002,570641);
/*!40000 ALTER TABLE `state_density` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-04-09 16:55:11
